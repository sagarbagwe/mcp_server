import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import json
import subprocess
import os
import io
import threading
import queue
import time

# --- Configuration ---
# Set your Google API Key (replace with your actual key or ensure it's in environment variables)
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key as an environment variable.")
    exit(1)

# Initialize the Gemini model for function calling
model = genai.GenerativeModel('gemini-2.5-pro')

# --- MCP Server Management ---
# Global variable to hold the MCP server process
mcp_server_process = None
# Queue for reading stdout from the MCP server asynchronously
stdout_queue = queue.Queue()
# Event to signal the stdout reader thread to stop
stop_stdout_reader_event = threading.Event()

def enqueue_output(out, queue, stop_event):
    """Reads lines from a stream and puts them into a queue."""
    for line in iter(out.readline, ''):
        if stop_event.is_set():
            break
        queue.put(line)
    out.close()

def start_mcp_server():
    """Starts the coda-mcp server as a subprocess."""
    global mcp_server_process
    print("Starting coda-mcp server...")
    try:
        # Use npx to run the latest coda-mcp. Ensure API_KEY is set in your environment.
        # The coda-mcp server communicates via stdio.
        mcp_server_process = subprocess.Popen(
            ["npx", "-y", "coda-mcp@latest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, # Important for text-based communication
            bufsize=1 # Line-buffered output
        )
        print("coda-mcp server started.")

        # Start a separate thread to read stdout to avoid deadlocks
        stop_stdout_reader_event.clear()
        threading.Thread(target=enqueue_output, args=(mcp_server_process.stdout, stdout_queue, stop_stdout_reader_event)).start()
        # You might want to also read stderr for debugging if needed.

        # Give the server a moment to initialize
        time.sleep(2)
        return True
    except FileNotFoundError:
        print("Error: 'npx' command not found. Make sure Node.js and npm are installed and in your PATH.")
        return False
    except Exception as e:
        print(f"Error starting coda-mcp server: {e}")
        return False

def stop_mcp_server():
    """Stops the coda-mcp server subprocess."""
    global mcp_server_process
    if mcp_server_process:
        print("Stopping coda-mcp server...")
        stop_stdout_reader_event.set() # Signal the reader thread to stop
        mcp_server_process.terminate() # or .kill() if terminate doesn't work
        mcp_server_process.wait(timeout=5) # Wait for it to terminate
        print("coda-mcp server stopped.")
    mcp_server_process = None

def call_mcp_tool(tool_name: str, **kwargs):
    """
    Generic function to send an MCP tool call request to the coda-mcp server
    and receive its response.
    """
    if not mcp_server_process or mcp_server_process.poll() is not None:
        print("MCP server not running or crashed. Attempting to restart...")
        if not start_mcp_server():
            return {"error": "Failed to restart MCP server."}

    request_id = str(time.time_ns()) # Unique ID for the request

    # Construct the MCP tool_call request payload
    mcp_request = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": kwargs
        }
    }
    request_json = json.dumps(mcp_request) + "\n"

    try:
        # Write the request to stdin of the MCP server
        mcp_server_process.stdin.write(request_json)
        mcp_server_process.stdin.flush()
        print(f"Sent MCP request to {tool_name}: {request_json.strip()}")

        # Read responses from stdout until we find our request_id or timeout
        response_data = None
        start_time = time.time()
        timeout = 30 # seconds
        while time.time() - start_time < timeout:
            try:
                line = stdout_queue.get(timeout=1) # Get with a timeout
                if line.strip():  # Only process non-empty lines
                    response = json.loads(line.strip())
                    if response.get("id") == request_id:
                        response_data = response
                        break
            except queue.Empty:
                # No response yet, continue waiting
                pass
            except json.JSONDecodeError as e:
                print(f"Warning: Non-JSON output from MCP server: {line.strip()}")
                # Continue trying to read more lines
            except Exception as e:
                print(f"Error processing MCP server output: {e}")
                break

        if response_data:
            if "result" in response_data:
                return response_data["result"]
            elif "error" in response_data:
                error_detail = response_data["error"]
                if isinstance(error_detail, dict):
                    return {"error": error_detail.get("message", str(error_detail))}
                else:
                    return {"error": str(error_detail)}
            else:
                return {"error": "MCP server response missing 'result' or 'error' key."}
        else:
            print(f"Timeout or no response for request {request_id} from MCP server.")
            # Try to read any stderr output for debugging
            try:
                # Non-blocking read of stderr
                if mcp_server_process.stderr.readable():
                    stderr_data = mcp_server_process.stderr.read()
                    if stderr_data:
                        print(f"MCP Server Stderr: {stderr_data}")
            except:
                pass
            return {"error": "MCP server did not respond in time or provided invalid output."}

    except Exception as e:
        print(f"Error communicating with MCP server: {e}")
        return {"error": f"Communication error with MCP server: {e}"}

# --- Tool Definitions for Gemini ---

# Define function declarations for Gemini
coda_list_documents_func = FunctionDeclaration(
    name="coda_list_documents",
    description="Lists all documents available to the authenticated user in Coda.io",
    parameters={}
)

coda_list_pages_func = FunctionDeclaration(
    name="coda_list_pages",
    description="Lists all pages within a Coda document",
    parameters={
        "type": "object",
        "properties": {
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        }
    }
)

coda_create_page_func = FunctionDeclaration(
    name="coda_create_page",
    description="Creates a new page in a Coda document",
    parameters={
        "type": "object",
        "properties": {
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"},
            "pageName": {"type": "string", "description": "The name of the new page"},
            "content": {"type": "string", "description": "Initial markdown content for the page"},
            "parentPageId": {"type": "string", "description": "Parent page ID to create a subpage"}
        },
        "required": ["pageName"]
    }
)

coda_get_page_content_func = FunctionDeclaration(
    name="coda_get_page_content",
    description="Retrieves the content of a Coda page as markdown",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        }
    }
)

coda_replace_page_content_func = FunctionDeclaration(
    name="coda_replace_page_content",
    description="Replaces a page's content with new markdown",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "content": {"type": "string", "description": "New markdown content for the page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["content"]
    }
)

coda_append_page_content_func = FunctionDeclaration(
    name="coda_append_page_content",
    description="Adds new markdown content to the end of a Coda page",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "content": {"type": "string", "description": "Markdown content to append"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["content"]
    }
)

coda_duplicate_page_func = FunctionDeclaration(
    name="coda_duplicate_page",
    description="Creates a copy of an existing Coda page with a new name",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "newPageName": {"type": "string", "description": "Name for the duplicated page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["newPageName"]
    }
)

coda_rename_page_func = FunctionDeclaration(
    name="coda_rename_page",
    description="Renames an existing Coda page",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The current page name"},
            "newName": {"type": "string", "description": "New name for the page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["newName"]
    }
)

coda_peek_page_func = FunctionDeclaration(
    name="coda_peek_page",
    description="Returns the first few lines of a Coda page as markdown",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "numLines": {"type": "integer", "description": "Number of lines to peek (default: 5)"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        }
    }
)

# Create the tools object
coda_tools = Tool(function_declarations=[
    coda_list_documents_func,
    coda_list_pages_func,
    coda_create_page_func,
    coda_get_page_content_func,
    coda_replace_page_content_func,
    coda_append_page_content_func,
    coda_duplicate_page_func,
    coda_rename_page_func,
    coda_peek_page_func
])

# --- Tool execution functions ---

def execute_coda_tool(tool_name: str, **kwargs):
    """Execute a coda tool and return the result."""
    # Map function calls to MCP tool names (they should be the same)
    tool_mapping = {
        "coda_list_documents": "coda_list_documents",
        "coda_list_pages": "coda_list_pages", 
        "coda_create_page": "coda_create_page",
        "coda_get_page_content": "coda_get_page_content",
        "coda_replace_page_content": "coda_replace_page_content",
        "coda_append_page_content": "coda_append_page_content",
        "coda_duplicate_page": "coda_duplicate_page",
        "coda_rename_page": "coda_rename_page",
        "coda_peek_page": "coda_peek_page"
    }
    
    mcp_tool_name = tool_mapping.get(tool_name)
    if not mcp_tool_name:
        return {"error": f"Unknown tool: {tool_name}"}
    
    # Clean up arguments - remove None values and handle special cases
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Handle special argument mapping for create_page
    if tool_name == "coda_create_page" and "pageName" in clean_kwargs:
        clean_kwargs["name"] = clean_kwargs.pop("pageName")
    
    # Handle default values
    if tool_name == "coda_peek_page" and "numLines" not in clean_kwargs:
        clean_kwargs["numLines"] = 5
    
    print(f"Tool Call: {tool_name}({clean_kwargs})")
    result = call_mcp_tool(mcp_tool_name, **clean_kwargs)
    print(f"Tool Result: {result}")
    return result

# --- Main Agent Loop ---

def run_agent_conversation(user_prompt: str):
    """
    Runs a conversation with the Gemini agent, allowing it to use coda-mcp tools.
    """
    # Start the chat with function calling enabled
    chat = model.start_chat()
    print(f"User: {user_prompt}")

    # Send the initial message to Gemini with tools
    response = chat.send_message(user_prompt, tools=[coda_tools])

    # Loop to handle tool calls and responses
    while True:
        if response.candidates[0].content.parts:
            # Check for function calls
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call)
            
            if function_calls:
                function_responses = []
                for function_call in function_calls:
                    tool_name = function_call.name
                    tool_args = dict(function_call.args)
                    
                    print(f"\nAgent suggests tool call: {tool_name} with args {tool_args}")
                    
                    try:
                        tool_output = execute_coda_tool(tool_name, **tool_args)
                        print(f"Tool output: {tool_output}")
                        
                        # Create function response
                        function_response = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response=tool_output
                            )
                        )
                        function_responses.append(function_response)
                        
                    except Exception as e:
                        error_message = f"Error executing tool {tool_name}: {e}"
                        print(error_message)
                        function_response = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"error": error_message}
                            )
                        )
                        function_responses.append(function_response)
                
                # Send all function responses back to the model
                response = chat.send_message(function_responses)
            else:
                # If no function calls, it's a text response, so print it and break the loop
                print("\nGemini's final response:")
                print(response.text)
                break
        else:
            print("\nGemini's final response:")
            print(response.text)
            break

# --- Execution ---
if __name__ == "__main__":
    if start_mcp_server():
        try:
            # Example Test Scenarios:

            # 1. List all Coda documents
            # run_agent_conversation("List all my Coda documents.")

            # 2. List pages in a specific document (replace with an actual document name or ID)
            # run_agent_conversation("List all pages in the document named 'My Project Notes'.")
            # run_agent_conversation("List pages in document with ID 'docId123'.") # If you know an ID

            # 3. Create a new page
            # Be careful with creation, as it will modify your Coda workspace!
            # run_agent_conversation("Create a new page called 'Agent Test Page' in the document 'My Project Notes' with the content 'Hello from Gemini agent!'.")

            # 4. Get content of a page
            # run_agent_conversation("Get the content of the page named 'Agent Test Page' in 'My Project Notes'.")
            
            # 5. Append content to a page
            # run_agent_conversation("Append 'This is some appended content.' to the page 'Agent Test Page' in 'My Project Notes'.")

            # 6. More complex chain of actions
            run_agent_conversation("""
            First, list all my Coda documents.
            Then, in the document named 'Meeting Minutes', create a new page called 'Action Items for July 22, 2025'.
            After that, add the following content to the new page:
            "- Task 1: Follow up on client email
            - Task 2: Prepare presentation slides"
            Finally, get the content of that newly created page.
            """)

        except Exception as e:
            print(f"An error occurred during agent conversation: {e}")
        finally:
            stop_mcp_server()
    else:
        print("Could not start MCP server, agent cannot function.")
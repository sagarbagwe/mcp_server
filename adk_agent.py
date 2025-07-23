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
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# --- MCP Server Management ---
# Global variable to hold the MCP server process
mcp_server_process = None
# Queue for reading stdout from the MCP server asynchronously
stdout_queue = queue.Queue()
# Event to signal the stdout reader thread to stop
stop_stdout_reader_event = threading.Event()

def enqueue_output(out, queue, stop_event):
    """Reads lines from a stream and puts them into a queue."""
    try:
        for line in iter(out.readline, ''):
            if stop_event.is_set():
                break
            queue.put(line)
    except Exception as e:
        print(f"Error in stdout reader: {e}")
    finally:
        try:
            out.close()
        except:
            pass

def start_mcp_server():
    """Starts the coda-mcp server as a subprocess."""
    global mcp_server_process
    print("Starting coda-mcp server...")
    try:
        # Use npx to run the latest coda-mcp. Ensure API_KEY is set in your environment.
        env = os.environ.copy()
        # Ensure the Coda API key is available to the MCP server
        if "API_KEY" not in env and "CODA_API_KEY" in env:
            env["API_KEY"] = env["CODA_API_KEY"]
        
        mcp_server_process = subprocess.Popen(
            ["npx", "-y", "coda-mcp@latest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        print("coda-mcp server started.")

        # Start a separate thread to read stdout to avoid deadlocks
        stop_stdout_reader_event.clear()
        threading.Thread(target=enqueue_output, args=(mcp_server_process.stdout, stdout_queue, stop_stdout_reader_event), daemon=True).start()

        # Give the server a moment to initialize
        time.sleep(3)
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
        stop_stdout_reader_event.set()
        mcp_server_process.terminate()
        try:
            mcp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("coda-mcp server did not terminate, killing...")
            mcp_server_process.kill()
            mcp_server_process.wait()
        print("coda-mcp server stopped.")
    mcp_server_process = None

def execute_mcp_tool(tool_name: str, **kwargs):
    """
    Generic function to send an MCP tool call request to the coda-mcp server
    and receive its response.
    """
    if not mcp_server_process or mcp_server_process.poll() is not None:
        print("MCP server not running or crashed. Attempting to restart...")
        if not start_mcp_server():
            return {"error": "Failed to restart MCP server."}

    request_id = str(time.time_ns())

    # Clean up kwargs - remove None values and empty strings
    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}

    # Construct the MCP tool_call request payload
    mcp_request = {
        "id": request_id,
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": clean_kwargs
        }
    }
    request_json = json.dumps(mcp_request) + "\n"

    try:
        # Clear any pending output first
        while not stdout_queue.empty():
            try:
                stdout_queue.get_nowait()
            except queue.Empty:
                break

        # Write the request to stdin of the MCP server
        mcp_server_process.stdin.write(request_json)
        mcp_server_process.stdin.flush()
        print(f"Sent MCP request to {tool_name}: {json.dumps(clean_kwargs, indent=2)}")

        # Read responses from stdout until we find our request_id or timeout
        response_data = None
        start_time = time.time()
        timeout = 45
        
        while time.time() - start_time < timeout:
            try:
                line = stdout_queue.get(timeout=2)
                line = line.strip()
                if line:
                    print(f"MCP Server output: {line}")
                    try:
                        response = json.loads(line)
                        if response.get("id") == request_id:
                            response_data = response
                            break
                    except json.JSONDecodeError as e:
                        print(f"Warning: Non-JSON output from MCP server: {line} - Error: {e}")
                        continue
            except queue.Empty:
                continue
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
            return {"error": "MCP server did not respond in time or provided invalid output."}

    except Exception as e:
        print(f"Error communicating with MCP server: {e}")
        return {"error": f"Communication error with MCP server: {e}"}

# --- Fixed Tool Definitions for Gemini (Using camelCase for MCP compatibility) ---

coda_list_documents_func = FunctionDeclaration(
    name="coda_list_documents",
    description="Lists all documents available to the authenticated user in Coda.io. Returns a list of documents with their IDs, names, and metadata.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    }
)

coda_list_pages_func = FunctionDeclaration(
    name="coda_list_pages",
    description="Lists all pages within a Coda document. Requires either docId or docName to identify the target document.",
    parameters={
        "type": "object",
        "properties": {
            "docId": {
                "type": "string", 
                "description": "The unique document ID (preferred for accuracy)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId)"
            }
        }
    }
)

coda_create_page_func = FunctionDeclaration(
    name="coda_create_page",
    description="Creates a new page in a Coda document. Requires pageName and either docId or docName.",
    parameters={
        "type": "object",
        "properties": {
            "docId": {
                "type": "string", 
                "description": "The unique document ID (preferred for accuracy)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId)"
            },
            "pageName": {
                "type": "string", 
                "description": "The name of the new page to create"
            },
            "content": {
                "type": "string", 
                "description": "Initial markdown content for the page (optional)"
            },
            "parentPageId": {
                "type": "string", 
                "description": "Parent page ID to create a subpage (optional)"
            }
        },
        "required": ["pageName"]
    }
)

coda_get_page_content_func = FunctionDeclaration(
    name="coda_get_page_content",
    description="Retrieves the complete content of a Coda page as markdown. Requires either pageId or pageName.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {
                "type": "string", 
                "description": "The unique page ID (preferred for accuracy)"
            },
            "pageName": {
                "type": "string", 
                "description": "The page name (alternative to pageId)"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required when using pageName)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId when using pageName)"
            }
        }
    }
)

coda_replace_page_content_func = FunctionDeclaration(
    name="coda_replace_page_content",
    description="Completely replaces a page's content with new markdown. WARNING: This will overwrite all existing content.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {
                "type": "string", 
                "description": "The unique page ID (preferred for accuracy)"
            },
            "pageName": {
                "type": "string", 
                "description": "The page name (alternative to pageId)"
            },
            "content": {
                "type": "string", 
                "description": "New markdown content to replace the entire page content"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required when using pageName)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId when using pageName)"
            }
        },
        "required": ["content"]
    }
)

coda_append_page_content_func = FunctionDeclaration(
    name="coda_append_page_content",
    description="Adds new markdown content to the end of a Coda page, preserving existing content.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {
                "type": "string", 
                "description": "The unique page ID (preferred for accuracy)"
            },
            "pageName": {
                "type": "string", 
                "description": "The page name (alternative to pageId)"
            },
            "content": {
                "type": "string", 
                "description": "Markdown content to append to the end of the page"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required when using pageName)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId when using pageName)"
            }
        },
        "required": ["content"]
    }
)

coda_peek_page_func = FunctionDeclaration(
    name="coda_peek_page",
    description="Returns the first few lines of a Coda page as markdown, useful for quickly checking page content.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {
                "type": "string", 
                "description": "The unique page ID (preferred for accuracy)"
            },
            "pageName": {
                "type": "string", 
                "description": "The page name (alternative to pageId)"
            },
            "numLines": {
                "type": "integer", 
                "description": "Number of lines to peek (default: 5, recommended max: 20)"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required when using pageName)"
            },
            "docName": {
                "type": "string", 
                "description": "The document name (alternative to docId when using pageName)"
            }
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
    coda_peek_page_func
])

# --- Fixed Tool execution functions (Using camelCase for MCP compatibility) ---

def coda_list_documents():
    """Wrapper for the coda_list_documents MCP tool."""
    return execute_mcp_tool("coda_list_documents")

def coda_list_pages(docId: str = None, docName: str = None):
    """Wrapper for the coda_list_pages MCP tool."""
    if not docId and not docName:
        return {"error": "coda_list_pages: Either docId or docName must be provided."}
    
    params = {}
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_list_pages", **params)

def coda_create_page(docId: str = None, docName: str = None, pageName: str = None, content: str = "", parentPageId: str = None):
    """Wrapper for the coda_create_page MCP tool."""
    if not pageName:
        return {"error": "coda_create_page: 'pageName' is required."}
    if not docId and not docName:
        return {"error": "coda_create_page: Either docId or docName must be provided."}
    
    params = {"pageName": pageName}
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    if content:
        params["content"] = content
    if parentPageId:
        params["parentPageId"] = parentPageId
    
    return execute_mcp_tool("coda_create_page", **params)

def coda_get_page_content(pageId: str = None, pageName: str = None, docId: str = None, docName: str = None):
    """Wrapper for the coda_get_page_content MCP tool."""
    if not pageId and not pageName:
        return {"error": "coda_get_page_content: Either pageId or pageName must be provided."}
    
    params = {}
    if pageId:
        params["pageId"] = pageId
    if pageName:
        params["pageName"] = pageName
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_get_page_content", **params)

def coda_replace_page_content(pageId: str = None, pageName: str = None, content: str = None, docId: str = None, docName: str = None):
    """Wrapper for the coda_replace_page_content MCP tool."""
    if content is None:
        return {"error": "coda_replace_page_content: 'content' is required."}
    if not pageId and not pageName:
        return {"error": "coda_replace_page_content: Either pageId or pageName must be provided."}
    
    params = {"content": content}
    if pageId:
        params["pageId"] = pageId
    if pageName:
        params["pageName"] = pageName
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_replace_page_content", **params)

def coda_append_page_content(pageId: str = None, pageName: str = None, content: str = None, docId: str = None, docName: str = None):
    """Wrapper for the coda_append_page_content MCP tool."""
    if content is None:
        return {"error": "coda_append_page_content: 'content' is required."}
    if not pageId and not pageName:
        return {"error": "coda_append_page_content: Either pageId or pageName must be provided."}
    
    params = {"content": content}
    if pageId:
        params["pageId"] = pageId
    if pageName:
        params["pageName"] = pageName
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_append_page_content", **params)

def coda_peek_page(pageId: str = None, pageName: str = None, numLines: int = 5, docId: str = None, docName: str = None):
    """Wrapper for the coda_peek_page MCP tool."""
    if not pageId and not pageName:
        return {"error": "coda_peek_page: Either pageId or pageName must be provided."}
    
    params = {"numLines": numLines}
    if pageId:
        params["pageId"] = pageId
    if pageName:
        params["pageName"] = pageName
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_peek_page", **params)

# --- Main Agent Loop ---

def run_interactive_agent_chat():
    """
    Runs an interactive chat with the Gemini agent, allowing it to use coda-mcp tools.
    """
    # Create a dictionary to map tool names to their corresponding Python functions
    available_tool_functions = {
        "coda_list_documents": coda_list_documents,
        "coda_list_pages": coda_list_pages,
        "coda_create_page": coda_create_page,
        "coda_get_page_content": coda_get_page_content,
        "coda_replace_page_content": coda_replace_page_content,
        "coda_append_page_content": coda_append_page_content,
        "coda_peek_page": coda_peek_page
    }

    # Start the MCP server before starting the chat
    if not start_mcp_server():
        print("Failed to start MCP server. Exiting.")
        return

    # Start the chat
    chat = model.start_chat()
    print("--- Enhanced Coda Agent Chat ---")
    print("Type 'exit' to end the chat.")
    print("Available commands:")
    print("- List documents and pages")
    print("- Create, read, update pages")
    print("- Manage page content with markdown")
    print("- Peek at page content")

    try:
        while True:
            user_prompt = input("\nYou: ")
            if user_prompt.lower() == 'exit':
                break

            print("Agent thinking...")
            try:
                # Send the message to Gemini with the defined tools
                response = chat.send_message(user_prompt, tools=[coda_tools])

                # Process Gemini's response
                while True:
                    if not response.candidates:
                        print("\nAgent: No candidates in response from Gemini.")
                        break

                    candidate_content = response.candidates[0].content
                    if not candidate_content:
                        print("\nAgent: No content in candidate response from Gemini.")
                        break

                    function_calls_found = []
                    text_parts_found = []

                    # Iterate through all parts to find function calls and text
                    for part in candidate_content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls_found.append(part.function_call)
                        if hasattr(part, 'text') and part.text:
                            text_parts_found.append(part.text)

                    if function_calls_found:
                        # If function calls are found, execute them
                        function_responses = []
                        for function_call in function_calls_found:
                            tool_name = function_call.name
                            tool_args = dict(function_call.args)

                            print(f"\nAgent suggests tool call: {tool_name} with args {tool_args}")
                            
                            # Execute the tool
                            if tool_name in available_tool_functions:
                                try:
                                    tool_output = available_tool_functions[tool_name](**tool_args)
                                    if not isinstance(tool_output, (dict, str)):
                                        tool_output = str(tool_output)
                                    print(f"Tool output: {tool_output}")
                                    function_responses.append(genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response=tool_output
                                        )
                                    ))
                                except Exception as e:
                                    error_message = f"Error executing tool {tool_name}: {e}"
                                    print(error_message)
                                    function_responses.append(genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"error": error_message}
                                        )
                                    ))
                            else:
                                error_message = f"Agent attempted to call unknown tool: {tool_name}"
                                print(error_message)
                                function_responses.append(genai.protos.Part(
                                    function_response=genai.protos.FunctionResponse(
                                        name=tool_name,
                                        response={"error": error_message}
                                    )
                                ))
                        
                        # Send the tool responses back to Gemini for further reasoning
                        response = chat.send_message(function_responses)
                    elif text_parts_found:
                        # If there are text parts and no function calls, it's the final text response
                        print("\nAgent:", "".join(text_parts_found))
                        break
                    else:
                        print("\nAgent: I received a response but couldn't interpret it (neither text nor tool calls).")
                        break

            except Exception as e:
                print(f"\nAn error occurred during agent conversation: {e}")
                break

    finally:
        # Clean up: stop the MCP server
        stop_mcp_server()

# --- Execution ---
if __name__ == "__main__":
    # Check for required environment variables
    if "API_KEY" not in os.environ and "CODA_API_KEY" not in os.environ:
        print("Warning: Neither API_KEY nor CODA_API_KEY environment variable is set for Coda MCP server.")
        print("Please set your Coda API key as an environment variable named 'API_KEY' or 'CODA_API_KEY'.")
    
    run_interactive_agent_chat()
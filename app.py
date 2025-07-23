import streamlit as st
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import json
import subprocess
import os
import io
import threading
import queue
import time
from datetime import datetime

# --- Configuration ---
# Set your Google API Key (replace with your actual key or ensure it's in environment variables)
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    st.error("Error: GOOGLE_API_KEY environment variable not set.")
    st.error("Please set your API key as an environment variable.")
    st.stop()

# Initialize the Gemini model for function calling
# 'gemini-pro' is generally good for text and function calling. 'gemini-2.5-pro' is a valid choice if available.
model = genai.GenerativeModel('gemini-2.5-pro')

# --- MCP Server Management ---
# Global variable to hold the MCP server process
if 'mcp_server_process' not in st.session_state:
    st.session_state.mcp_server_process = None
if 'stdout_queue' not in st.session_state:
    st.session_state.stdout_queue = queue.Queue()
if 'stop_stdout_reader_event' not in st.session_state:
    st.session_state.stop_stdout_reader_event = threading.Event()

def enqueue_output(out, queue, stop_event):
    """Reads lines from a stream and puts them into a queue."""
    for line in iter(out.readline, ''):
        if stop_event.is_set():
            break
        queue.put(line)
    out.close()
def start_mcp_server():
    """Starts the coda-mcp server as a subprocess and sets up stdout reading."""
    print("Starting coda-mcp server...")

    try:
        # Launch the MCP server using npx
        # CODA_API_KEY is injected from environment variables
        mcp_server_process = subprocess.Popen(
            ["npx", "-y", "coda-mcp@latest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "CODA_API_KEY": os.environ.get("CODA_API_KEY", "")}
        )

        print("coda-mcp server started.")

        # Clear the stop event in case this is a restart
        st.session_state.stop_stdout_reader_event.clear()

        # Start a separate thread to continuously read stdout
        threading.Thread(
            target=enqueue_output,
            args=(
                mcp_server_process.stdout,
                st.session_state.stdout_queue,
                st.session_state.stop_stdout_reader_event,
            ),
            daemon=True
        ).start()

        # Optional: You can start another thread to read stderr for logging/debugging

        # Give the server a second to initialize
        time.sleep(2)

        # Save the process handle in Streamlit session state
        st.session_state.mcp_server_process = mcp_server_process
        return True

    except FileNotFoundError:
        st.error("Error: 'npx' command not found. Make sure Node.js and npm are installed and in your PATH.")
        return False

    except Exception as e:
        st.error(f"Error starting coda-mcp server: {e}")
        return False

def stop_mcp_server():
    """Stops the coda-mcp server subprocess."""
    if st.session_state.mcp_server_process:
        print("Stopping coda-mcp server...")
        st.session_state.stop_stdout_reader_event.set() # Signal the reader thread to stop
        st.session_state.mcp_server_process.terminate() # or .kill() if terminate doesn't work
        try:
            st.session_state.mcp_server_process.wait(timeout=5) # Wait for it to terminate
        except subprocess.TimeoutExpired:
            print("coda-mcp server did not terminate, killing...")
            st.session_state.mcp_server_process.kill()
            st.session_state.mcp_server_process.wait()
        print("coda-mcp server stopped.")
    st.session_state.mcp_server_process = None

def execute_mcp_tool(tool_name: str, **kwargs):
    """
    Generic function to send an MCP tool call request to the coda-mcp server
    and receive its response. Renamed from call_mcp_tool to avoid confusion with Gemini's tool calls.
    """
    if not st.session_state.mcp_server_process or st.session_state.mcp_server_process.poll() is not None:
        st.warning("MCP server not running or crashed. Attempting to restart...")
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
        st.session_state.mcp_server_process.stdin.write(request_json)
        st.session_state.mcp_server_process.stdin.flush()
        print(f"Sent MCP request to {tool_name}: {request_json.strip()}")

        # Read responses from stdout until we find our request_id or timeout
        response_data = None
        start_time = time.time()
        timeout = 30 # seconds
        while time.time() - start_time < timeout:
            try:
                line = st.session_state.stdout_queue.get(timeout=1) # Get with a timeout
                if line.strip():  # Only process non-empty lines
                    response = json.loads(line.strip())
                    if response.get("id") == request_id:
                        response_data = response
                        break
            except queue.Empty:
                # No response yet, continue waiting
                pass
            except json.JSONDecodeError as e:
                print(f"Warning: Non-JSON output from MCP server: {line.strip()} - Error: {e}")
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
                if st.session_state.mcp_server_process.stderr.readable():
                    stderr_data = st.session_state.mcp_server_process.stderr.read()
                    if stderr_data:
                        print(f"MCP Server Stderr: {stderr_data}")
            except Exception as e_stderr:
                print(f"Error reading stderr: {e_stderr}")
            return {"error": "MCP server did not respond in time or provided invalid output."}

    except Exception as e:
        print(f"Error communicating with MCP server: {e}")
        return {"error": f"Communication error with MCP server: {e}"}

# --- Tool Definitions for Gemini (FunctionDeclarations) ---
# Removed 'anyOf' as it's not supported directly in the FunctionDeclaration schema.
# Input validation will now happen in the Python wrapper functions.

coda_list_documents_func = FunctionDeclaration(
    name="coda_list_documents",
    description="Lists all documents available to the authenticated user in Coda.io",
    parameters={
        "type": "object",
        "properties": {} # No parameters for this tool
    }
)

coda_list_pages_func = FunctionDeclaration(
    name="coda_list_pages",
    description="Lists all pages within a Coda document. Requires either documentId or documentName.",
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
    description="Creates a new page in a Coda document. Requires a pageName and either documentId or documentName. Optionally, initial markdown content can be provided, and a parentPageId to create a subpage.",
    parameters={
        "type": "object",
        "properties": {
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"},
            "pageName": {"type": "string", "description": "The name of the new page"},
            "content": {"type": "string", "description": "Initial markdown content for the page"},
            "parentPageId": {"type": "string", "description": "Parent page ID to create a subpage"}
        },
        "required": ["pageName"] # pageName is always required
    }
)

coda_get_page_content_func = FunctionDeclaration(
    name="coda_get_page_content",
    description="Retrieves the content of a Coda page as markdown. Requires either pageId or pageName. If using pageName, you should also provide documentId or documentName for disambiguation.",
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
    description="Replaces a page's content with new markdown. Requires content and either pageId or pageName. If using pageName, you should also provide documentId or documentName for disambiguation.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "content": {"type": "string", "description": "New markdown content for the page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["content"] # Content is always required
    }
)

coda_append_page_content_func = FunctionDeclaration(
    name="coda_append_page_content",
    description="Adds new markdown content to the end of a Coda page. Requires content and either pageId or pageName. If using pageName, you should also provide documentId or documentName for disambiguation.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "content": {"type": "string", "description": "Markdown content to append"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["content"] # Content is always required
    }
)

coda_duplicate_page_func = FunctionDeclaration(
    name="coda_duplicate_page",
    description="Creates a copy of an existing Coda page with a new name. Requires newPageName and either pageId or pageName. If using pageName, you should also provide documentId or documentName for disambiguation.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The page name"},
            "newPageName": {"type": "string", "description": "Name for the duplicated page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["newPageName"] # newPageName is always required
    }
)

coda_rename_page_func = FunctionDeclaration(
    name="coda_rename_page",
    description="Renames an existing Coda page. Requires newName and either pageId or pageName. If using pageName, you should also provide documentId or documentName for disambiguation.",
    parameters={
        "type": "object",
        "properties": {
            "pageId": {"type": "string", "description": "The page ID"},
            "pageName": {"type": "string", "description": "The current page name"},
            "newName": {"type": "string", "description": "New name for the page"},
            "documentId": {"type": "string", "description": "The document ID"},
            "documentName": {"type": "string", "description": "The document name"}
        },
        "required": ["newName"] # newName is always required
    }
)

coda_peek_page_func = FunctionDeclaration(
    name="coda_peek_page",
    description="Returns the first few lines of a Coda page as markdown. Requires either pageId or pageName. Optionally, specify the number of lines to peek. If using pageName, you should also provide documentId or documentName for disambiguation.",
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

# Create the tools object (a single Tool containing multiple FunctionDeclarations)
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

# --- Tool execution functions (Python wrappers) ---
# These functions will be called by your Python script based on Gemini's suggestions.
# Their names must match the 'name' in FunctionDeclaration.

def coda_list_documents():
    """Wrapper for the coda_list_documents MCP tool."""
    return execute_mcp_tool("coda_list_documents")

def coda_list_pages(documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_list_pages MCP tool.
    Adds validation for documentId or documentName.
    """
    if not documentId and not documentName:
        return {"error": "coda_list_pages: Either documentId or documentName must be provided."}
    return execute_mcp_tool("coda_list_pages", documentId=documentId, documentName=documentName)

def coda_create_page(documentId: str = None, documentName: str = None, pageName: str = None, content: str = "", parentPageId: str = None):
    """
    Wrapper for the coda_create_page MCP tool.
    Adds validation for pageName and documentId/documentName.
    """
    if not pageName:
        return {"error": "coda_create_page: 'pageName' is required."}
    if not documentId and not documentName:
        return {"error": "coda_create_page: Either documentId or documentName must be provided."}
    
    # MCP expects 'name' for page name, not 'pageName' in some contexts, adjust here.
    return execute_mcp_tool("coda_create_page", documentId=documentId, documentName=documentName, name=pageName, content=content, parentPageId=parentPageId)

def coda_get_page_content(pageId: str = None, pageName: str = None, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_get_page_content MCP tool.
    Adds validation for pageId or pageName.
    """
    if not pageId and not pageName:
        return {"error": "coda_get_page_content: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_get_page_content", pageId=pageId, pageName=pageName, documentId=documentId, documentName=documentName)

def coda_replace_page_content(pageId: str = None, pageName: str = None, content: str = None, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_replace_page_content MCP tool.
    Adds validation for content and pageId/pageName.
    """
    if content is None: # Explicitly check for None, empty string is valid content
        return {"error": "coda_replace_page_content: 'content' is required."}
    if not pageId and not pageName:
        return {"error": "coda_replace_page_content: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_replace_page_content", pageId=pageId, pageName=pageName, content=content, documentId=documentId, documentName=documentName)

def coda_append_page_content(pageId: str = None, pageName: str = None, content: str = None, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_append_page_content MCP tool.
    Adds validation for content and pageId/pageName.
    """
    if content is None: # Explicitly check for None, empty string is valid content
        return {"error": "coda_append_page_content: 'content' is required."}
    if not pageId and not pageName:
        return {"error": "coda_append_page_content: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_append_page_content", pageId=pageId, pageName=pageName, content=content, documentId=documentId, documentName=documentName)

def coda_duplicate_page(pageId: str = None, pageName: str = None, newPageName: str = None, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_duplicate_page MCP tool.
    Adds validation for newPageName and pageId/pageName.
    """
    if not newPageName:
        return {"error": "coda_duplicate_page: 'newPageName' is required."}
    if not pageId and not pageName:
        return {"error": "coda_duplicate_page: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_duplicate_page", pageId=pageId, pageName=pageName, newPageName=newPageName, documentId=documentId, documentName=documentName)

def coda_rename_page(pageId: str = None, pageName: str = None, newName: str = None, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_rename_page MCP tool.
    Adds validation for newName and pageId/pageName.
    """
    if not newName:
        return {"error": "coda_rename_page: 'newName' is required."}
    if not pageId and not pageName:
        return {"error": "coda_rename_page: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_rename_page", pageId=pageId, pageName=pageName, newName=newName, documentId=documentId, documentName=documentName)

def coda_peek_page(pageId: str = None, pageName: str = None, numLines: int = 5, documentId: str = None, documentName: str = None):
    """
    Wrapper for the coda_peek_page MCP tool.
    Adds validation for pageId or pageName.
    """
    if not pageId and not pageName:
        return {"error": "coda_peek_page: Either pageId or pageName must be provided."}
    return execute_mcp_tool("coda_peek_page", pageId=pageId, pageName=pageName, numLines=numLines, documentId=documentId, documentName=documentName)

# --- Streamlit UI ---

def main():
    st.set_page_config(
        page_title="Coda Agent Chat",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("🤖 Coda Agent Chat")
    st.markdown("Chat with an AI agent that can interact with your Coda documents!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat' not in st.session_state:
        st.session_state.chat = None
    if 'server_started' not in st.session_state:
        st.session_state.server_started = False
    
    # Sidebar for server management
    with st.sidebar:
        st.header("🔧 Server Management")
        
        if not st.session_state.server_started:
            if st.button("🚀 Start MCP Server", type="primary"):
                with st.spinner("Starting MCP server..."):
                    if start_mcp_server():
                        st.session_state.server_started = True
                        st.session_state.chat = model.start_chat()
                        st.success("MCP server started successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to start MCP server")
        else:
            st.success("✅ MCP Server is running")
            if st.button("🛑 Stop MCP Server", type="secondary"):
                stop_mcp_server()
                st.session_state.server_started = False
                st.session_state.chat = None
                st.info("MCP server stopped")
                st.rerun()
        
        st.divider()
        
        # Available tools info
        st.header("🛠️ Available Tools")
        tools_info = [
            "📋 List Documents",
            "📄 List Pages",
            "➕ Create Page",
            "📖 Get Page Content",
            "✏️ Replace Page Content",
            "📝 Append Page Content",
            "📋 Duplicate Page",
            "🏷️ Rename Page",
            "👀 Peek Page"
        ]
        for tool in tools_info:
            st.markdown(f"• {tool}")
        
        st.divider()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.server_started:
                st.session_state.chat = model.start_chat()
            st.rerun()
    
    # Main chat interface
    if not st.session_state.server_started:
        st.warning("⚠️ Please start the MCP server first using the sidebar.")
        st.info("💡 Make sure you have set the GOOGLE_API_KEY environment variable and have Node.js/npm installed.")
        return
    
    # Create dictionary to map tool names to their corresponding Python functions
    available_tool_functions = {
        "coda_list_documents": coda_list_documents,
        "coda_list_pages": coda_list_pages,
        "coda_create_page": coda_create_page,
        "coda_get_page_content": coda_get_page_content,
        "coda_replace_page_content": coda_replace_page_content,
        "coda_append_page_content": coda_append_page_content,
        "coda_duplicate_page": coda_duplicate_page,
        "coda_rename_page": coda_rename_page,
        "coda_peek_page": coda_peek_page
    }
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if "tool_calls" in message:
                    # Display tool calls
                    for tool_call in message["tool_calls"]:
                        with st.expander(f"🔧 Tool Call: {tool_call['name']}"):
                            st.code(json.dumps(tool_call["args"], indent=2), language="json")
                            if "output" in tool_call:
                                st.markdown("**Output:**")
                                if isinstance(tool_call["output"], dict):
                                    st.json(tool_call["output"])
                                else:
                                    st.markdown(str(tool_call["output"]))
                
                if "content" in message:
                    st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your Coda documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Agent thinking..."):
                try:
                    # Send the message to Gemini with the defined tools
                    response = st.session_state.chat.send_message(prompt, tools=[coda_tools])
                    
                    assistant_message = {"role": "assistant"}
                    tool_calls_made = []
                    
                    # Process Gemini's response
                    while True:
                        if not response.candidates:
                            st.error("No candidates in response from Gemini.")
                            break

                        candidate_content = response.candidates[0].content
                        if not candidate_content:
                            st.error("No content in candidate response from Gemini.")
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
                                tool_args = dict(function_call.args) # Convert to dictionary

                                st.info(f"🔧 Executing tool: **{tool_name}**")
                                
                                # Store tool call info
                                tool_call_info = {
                                    "name": tool_name,
                                    "args": tool_args
                                }
                                
                                # Execute the tool
                                if tool_name in available_tool_functions:
                                    try:
                                        tool_output = available_tool_functions[tool_name](**tool_args)
                                        tool_call_info["output"] = tool_output
                                        
                                        # Ensure tool_output is a dictionary or string for FunctionResponse
                                        if not isinstance(tool_output, (dict, str)):
                                            tool_output = str(tool_output)
                                        
                                        function_responses.append(genai.protos.Part(
                                            function_response=genai.protos.FunctionResponse(
                                                name=tool_name,
                                                response=tool_output
                                            )
                                        ))
                                    except Exception as e:
                                        error_message = f"Error executing tool {tool_name}: {e}"
                                        tool_call_info["output"] = {"error": error_message}
                                        st.error(error_message)
                                        function_responses.append(genai.protos.Part(
                                            function_response=genai.protos.FunctionResponse(
                                                name=tool_name,
                                                response={"error": error_message}
                                            )
                                        ))
                                else:
                                    error_message = f"Agent attempted to call unknown tool: {tool_name}"
                                    tool_call_info["output"] = {"error": error_message}
                                    st.error(error_message)
                                    function_responses.append(genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"error": error_message}
                                        )
                                    ))
                                
                                tool_calls_made.append(tool_call_info)
                            
                            # Send the tool responses back to Gemini for further reasoning
                            response = st.session_state.chat.send_message(function_responses)
                        elif text_parts_found:
                            # If there are text parts and no function calls, it's the final text response
                            final_response = "".join(text_parts_found)
                            st.markdown(final_response)
                            assistant_message["content"] = final_response
                            break # Exit inner loop, await next user input
                        else:
                            # This case should ideally not happen if response.candidates[0].content exists
                            st.error("I received a response but couldn't interpret it (neither text nor tool calls).")
                            break # Exit inner loop
                    
                    # Add tool calls to assistant message if any were made
                    if tool_calls_made:
                        assistant_message["tool_calls"] = tool_calls_made
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    st.error(f"An error occurred during agent conversation: {e}")

if __name__ == "__main__":
    main()
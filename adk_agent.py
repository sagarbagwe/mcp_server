
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import json
import subprocess
import os
import io
import threading
import queue
import time

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set your API key as an environment variable.")
    exit(1)

model = genai.GenerativeModel('gemini-2.5-pro')

mcp_server_process = None
stdout_queue = queue.Queue()
stop_stdout_reader_event = threading.Event()

def enqueue_output(out, queue, stop_event):
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
    global mcp_server_process
    print("Starting coda-mcp server...")
    try:
        env = os.environ.copy()
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

        stop_stdout_reader_event.clear()
        threading.Thread(target=enqueue_output, args=(mcp_server_process.stdout, stdout_queue, stop_stdout_reader_event), daemon=True).start()

        time.sleep(3)
        return True
    except FileNotFoundError:
        print("Error: 'npx' command not found. Make sure Node.js and npm are installed and in your PATH.")
        return False
    except Exception as e:
        print(f"Error starting coda-mcp server: {e}")
        return False

def stop_mcp_server():
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
    if not mcp_server_process or mcp_server_process.poll() is not None:
        print("MCP server not running or crashed. Attempting to restart...")
        if not start_mcp_server():
            return {"error": "Failed to restart MCP server."}

    request_id = str(time.time_ns())

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}

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
        while not stdout_queue.empty():
            try:
                stdout_queue.get_nowait()
            except queue.Empty:
                break

        mcp_server_process.stdin.write(request_json)
        mcp_server_process.stdin.flush()
        print(f"Sent MCP request to {tool_name}: {json.dumps(clean_kwargs, indent=2)}")

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
        },
        "required": []
    }
)

coda_create_page_func = FunctionDeclaration(
    name="coda_create_page",
    description="Creates a new page in a Coda document. Requires name and docId.",
    parameters={
        "type": "object",
        "properties": {
            "docId": {
                "type": "string", 
                "description": "The unique document ID (required)"
            },
            "name": {
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
        "required": ["docId", "name"]
    }
)

coda_get_page_content_func = FunctionDeclaration(
    name="coda_get_page_content",
    description="Retrieves the complete content of a Coda page as markdown. Requires pageIdOrName and docId.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string", 
                "description": "The page ID or page name"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required)"
            }
        },
        "required": ["pageIdOrName", "docId"]
    }
)

coda_replace_page_content_func = FunctionDeclaration(
    name="coda_replace_page_content",
    description="Completely replaces a page's content with new markdown. WARNING: This will overwrite all existing content.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string", 
                "description": "The page ID or page name"
            },
            "content": {
                "type": "string", 
                "description": "New markdown content to replace the entire page content"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required)"
            }
        },
        "required": ["pageIdOrName", "content", "docId"]
    }
)

coda_append_page_content_func = FunctionDeclaration(
    name="coda_append_page_content",
    description="Adds new markdown content to the end of a Coda page, preserving existing content.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string", 
                "description": "The page ID or page name"
            },
            "content": {
                "type": "string", 
                "description": "Markdown content to append to the end of the page"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required)"
            }
        },
        "required": ["pageIdOrName", "content", "docId"]
    }
)

coda_peek_page_func = FunctionDeclaration(
    name="coda_peek_page",
    description="Returns the first few lines of a Coda page as markdown, useful for quickly checking page content.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string", 
                "description": "The page ID or page name"
            },
            "numLines": {
                "type": "integer", 
                "description": "Number of lines to peek (default: 5, recommended max: 20)"
            },
            "docId": {
                "type": "string", 
                "description": "The document ID (required)"
            }
        },
        "required": ["pageIdOrName", "docId"]
    }
)

# New FunctionDeclaration for coda_duplicate_page
coda_duplicate_page_func = FunctionDeclaration(
    name="coda_duplicate_page",
    description="Creates a copy of an existing page with a new name. Requires pageIdOrName, newName, and docId.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string",
                "description": "The ID or name of the page to duplicate."
            },
            "newName": {
                "type": "string",
                "description": "The name for the new duplicated page."
            },
            "docId": {
                "type": "string",
                "description": "The document ID (required)."
            }
        },
        "required": ["pageIdOrName", "newName", "docId"]
    }
)

# New FunctionDeclaration for coda_rename_page
coda_rename_page_func = FunctionDeclaration(
    name="coda_rename_page",
    description="Renames an existing page. Requires pageIdOrName, newName, and docId.",
    parameters={
        "type": "object",
        "properties": {
            "pageIdOrName": {
                "type": "string",
                "description": "The ID or name of the page to rename."
            },
            "newName": {
                "type": "string",
                "description": "The new name for the page."
            },
            "docId": {
                "type": "string",
                "description": "The document ID (required)."
            }
        },
        "required": ["pageIdOrName", "newName", "docId"]
    }
)

# New FunctionDeclaration for coda_resolve_link
coda_resolve_link_func = FunctionDeclaration(
    name="coda_resolve_link",
    description="Resolves metadata given a browser link to a Coda object. Returns information about the linked object.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The Coda URL to resolve."
            }
        },
        "required": ["url"]
    }
)


coda_tools = Tool(function_declarations=[
    coda_list_documents_func,
    coda_list_pages_func,
    coda_create_page_func,
    coda_get_page_content_func,
    coda_replace_page_content_func,
    coda_append_page_content_func,
    coda_peek_page_func,
    coda_duplicate_page_func, # Added
    coda_rename_page_func,    # Added
    coda_resolve_link_func    # Added
])

def coda_list_documents():
    return execute_mcp_tool("coda_list_documents")

def coda_list_pages(docId: str = None, docName: str = None):
    params = {}
    if docId:
        params["docId"] = docId
    if docName:
        params["docName"] = docName
    
    return execute_mcp_tool("coda_list_pages", **params)

def coda_create_page(docId: str, name: str, content: str = "", parentPageId: str = None):
    params = {"docId": docId, "name": name}
    if content:
        params["content"] = content
    if parentPageId:
        params["parentPageId"] = parentPageId
    
    return execute_mcp_tool("coda_create_page", **params)

def coda_get_page_content(pageIdOrName: str, docId: str):
    params = {"pageIdOrName": pageIdOrName, "docId": docId}
    return execute_mcp_tool("coda_get_page_content", **params)

def coda_replace_page_content(pageIdOrName: str, content: str, docId: str):
    params = {"pageIdOrName": pageIdOrName, "content": content, "docId": docId}
    return execute_mcp_tool("coda_replace_page_content", **params)

def coda_append_page_content(pageIdOrName: str, content: str, docId: str):
    params = {"pageIdOrName": pageIdOrName, "content": content, "docId": docId}
    return execute_mcp_tool("coda_append_page_content", **params)

def coda_peek_page(pageIdOrName: str, docId: str, numLines: int = 5):
    params = {"pageIdOrName": pageIdOrName, "docId": docId, "numLines": numLines}
    return execute_mcp_tool("coda_peek_page", **params)

# New wrapper function for coda_duplicate_page
def coda_duplicate_page(pageIdOrName: str, newName: str, docId: str):
    params = {"pageIdOrName": pageIdOrName, "newName": newName, "docId": docId}
    return execute_mcp_tool("coda_duplicate_page", **params)

# New wrapper function for coda_rename_page
def coda_rename_page(pageIdOrName: str, newName: str, docId: str):
    params = {"pageIdOrName": pageIdOrName, "newName": newName, "docId": docId}
    return execute_mcp_tool("coda_rename_page", **params)

# New wrapper function for coda_resolve_link
def coda_resolve_link(url: str):
    params = {"url": url}
    return execute_mcp_tool("coda_resolve_link", **params)

def run_interactive_agent_chat():
    available_tool_functions = {
        "coda_list_documents": coda_list_documents,
        "coda_list_pages": coda_list_pages,
        "coda_create_page": coda_create_page,
        "coda_get_page_content": coda_get_page_content,
        "coda_replace_page_content": coda_replace_page_content,
        "coda_append_page_content": coda_append_page_content,
        "coda_peek_page": coda_peek_page,
        "coda_duplicate_page": coda_duplicate_page, # Added
        "coda_rename_page": coda_rename_page,       # Added
        "coda_resolve_link": coda_resolve_link      # Added
    }

    if not start_mcp_server():
        print("Failed to start MCP server. Exiting.")
        return

    chat = model.start_chat()
    print("--- Enhanced Coda Agent Chat ---")
    print("Type 'exit' to end the chat.")
    print("Available commands:")
    print("- List documents and pages")
    print("- Create, read, update pages")
    print("- Manage page content with markdown")
    print("- Peek at page content")
    print("- Duplicate and rename pages") # Added
    print("- Resolve Coda links")          # Added

    try:
        while True:
            user_prompt = input("\nYou: ")
            if user_prompt.lower() == 'exit':
                break

            print("Agent thinking...")
            try:
                response = chat.send_message(user_prompt, tools=[coda_tools])

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

                    for part in candidate_content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls_found.append(part.function_call)
                        if hasattr(part, 'text') and part.text:
                            text_parts_found.append(part.text)

                    if function_calls_found:
                        function_responses = []
                        for function_call in function_calls_found:
                            tool_name = function_call.name
                            tool_args = dict(function_call.args)

                            print(f"\nAgent suggests tool call: {tool_name} with args {tool_args}")
                            
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
                        
                        response = chat.send_message(function_responses)
                    elif text_parts_found:
                        print("\nAgent:", "".join(text_parts_found))
                        break
                    else:
                        print("\nAgent: I received a response but couldn't interpret it (neither text nor tool calls).")
                        break

            except Exception as e:
                print(f"\nAn error occurred during agent conversation: {e}")
                break

    finally:
        stop_mcp_server()

if __name__ == "__main__":
    if "API_KEY" not in os.environ and "CODA_API_KEY" not in os.environ:
        print("Warning: Neither API_KEY nor CODA_API_KEY environment variable is set for Coda MCP server.")
        print("Please set your Coda API key as an environment variable named 'API_KEY' or 'CODA_API_KEY'.")
    
    run_interactive_agent_chat()
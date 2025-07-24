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
import pickle
from datetime import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Coda AI Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve chat layout
st.markdown("""
<style>
    /* Hide the main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve chat message styling */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* Make chat input more prominent */
    .stChatInput > div > div > div {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    /* Improve sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Better spacing for chat container */
    .main .block-container {
        padding-bottom: 2rem;
    }
    
    /* Style for welcome message */
    .welcome-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "mcp_server_process" not in st.session_state:
    st.session_state.mcp_server_process = None
if "stdout_queue" not in st.session_state:
    st.session_state.stdout_queue = queue.Queue()
if "stop_stdout_reader_event" not in st.session_state:
    st.session_state.stop_stdout_reader_event = threading.Event()
if "model" not in st.session_state:
    st.session_state.model = None
if "session_name" not in st.session_state:
    st.session_state.session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create sessions directory
SESSIONS_DIR = Path("chat_sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

def enqueue_output(out, queue, stop_event):
    try:
        for line in iter(out.readline, ''):
            if stop_event.is_set():
                break
            queue.put(line)
    except Exception as e:
        st.error(f"Error in stdout reader: {e}")
    finally:
        try:
            out.close()
        except:
            pass

def start_mcp_server():
    if st.session_state.mcp_server_process is not None:
        return True
        
    try:
        env = os.environ.copy()
        if "API_KEY" not in env and "CODA_API_KEY" in env:
            env["API_KEY"] = env["CODA_API_KEY"]
        
        st.session_state.mcp_server_process = subprocess.Popen(
            ["npx", "-y", "coda-mcp@latest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )

        st.session_state.stop_stdout_reader_event.clear()
        threading.Thread(
            target=enqueue_output, 
            args=(st.session_state.mcp_server_process.stdout, st.session_state.stdout_queue, st.session_state.stop_stdout_reader_event), 
            daemon=True
        ).start()

        time.sleep(3)
        return True
    except FileNotFoundError:
        st.error("Error: 'npx' command not found. Make sure Node.js and npm are installed and in your PATH.")
        return False
    except Exception as e:
        st.error(f"Error starting coda-mcp server: {e}")
        return False

def stop_mcp_server():
    if st.session_state.mcp_server_process:
        st.session_state.stop_stdout_reader_event.set()
        st.session_state.mcp_server_process.terminate()
        try:
            st.session_state.mcp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            st.session_state.mcp_server_process.kill()
            st.session_state.mcp_server_process.wait()
    st.session_state.mcp_server_process = None

def execute_mcp_tool(tool_name: str, **kwargs):
    if not st.session_state.mcp_server_process or st.session_state.mcp_server_process.poll() is not None:
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
        while not st.session_state.stdout_queue.empty():
            try:
                st.session_state.stdout_queue.get_nowait()
            except queue.Empty:
                break

        st.session_state.mcp_server_process.stdin.write(request_json)
        st.session_state.mcp_server_process.stdin.flush()

        response_data = None
        start_time = time.time()
        timeout = 45
        
        while time.time() - start_time < timeout:
            try:
                line = st.session_state.stdout_queue.get(timeout=2)
                line = line.strip()
                if line:
                    try:
                        response = json.loads(line)
                        if response.get("id") == request_id:
                            response_data = response
                            break
                    except json.JSONDecodeError:
                        continue
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Error processing MCP server output: {e}")
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
            return {"error": "MCP server did not respond in time or provided invalid output."}

    except Exception as e:
        return {"error": f"Communication error with MCP server: {e}"}

# Function declarations for Coda tools
coda_tools_declarations = [
    FunctionDeclaration(
        name="coda_list_documents",
        description="Lists all documents available to the authenticated user in Coda.io.",
        parameters={"type": "object", "properties": {}, "required": []}
    ),
    FunctionDeclaration(
        name="coda_list_pages",
        description="Lists all pages within a Coda document.",
        parameters={
            "type": "object",
            "properties": {
                "docId": {"type": "string", "description": "The unique document ID (preferred)"},
                "docName": {"type": "string", "description": "The document name (alternative)"}
            },
            "required": []
        }
    ),
    FunctionDeclaration(
        name="coda_create_page",
        description="Creates a new page in a Coda document.",
        parameters={
            "type": "object",
            "properties": {
                "docId": {"type": "string", "description": "The unique document ID (required)"},
                "name": {"type": "string", "description": "The name of the new page"},
                "content": {"type": "string", "description": "Initial markdown content (optional)"},
                "parentPageId": {"type": "string", "description": "Parent page ID for subpage (optional)"}
            },
            "required": ["docId", "name"]
        }
    ),
    FunctionDeclaration(
        name="coda_get_page_content",
        description="Retrieves the complete content of a Coda page as markdown.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The page ID or page name"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_replace_page_content",
        description="Completely replaces a page's content with new markdown.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The page ID or page name"},
                "content": {"type": "string", "description": "New markdown content"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "content", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_append_page_content",
        description="Adds new markdown content to the end of a Coda page.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The page ID or page name"},
                "content": {"type": "string", "description": "Markdown content to append"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "content", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_peek_page",
        description="Returns the first few lines of a Coda page as markdown.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The page ID or page name"},
                "numLines": {"type": "integer", "description": "Number of lines to peek (default: 5)"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_duplicate_page",
        description="Creates a copy of an existing page with a new name.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The ID or name of the page to duplicate"},
                "newName": {"type": "string", "description": "The name for the new duplicated page"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "newName", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_rename_page",
        description="Renames an existing page.",
        parameters={
            "type": "object",
            "properties": {
                "pageIdOrName": {"type": "string", "description": "The ID or name of the page to rename"},
                "newName": {"type": "string", "description": "The new name for the page"},
                "docId": {"type": "string", "description": "The document ID (required)"}
            },
            "required": ["pageIdOrName", "newName", "docId"]
        }
    ),
    FunctionDeclaration(
        name="coda_resolve_link",
        description="Resolves metadata given a browser link to a Coda object.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The Coda URL to resolve"}
            },
            "required": ["url"]
        }
    )
]

coda_tools = Tool(function_declarations=coda_tools_declarations)

# Tool execution functions
available_tool_functions = {
    "coda_list_documents": lambda: execute_mcp_tool("coda_list_documents"),
    "coda_list_pages": lambda **kwargs: execute_mcp_tool("coda_list_pages", **kwargs),
    "coda_create_page": lambda **kwargs: execute_mcp_tool("coda_create_page", **kwargs),
    "coda_get_page_content": lambda **kwargs: execute_mcp_tool("coda_get_page_content", **kwargs),
    "coda_replace_page_content": lambda **kwargs: execute_mcp_tool("coda_replace_page_content", **kwargs),
    "coda_append_page_content": lambda **kwargs: execute_mcp_tool("coda_append_page_content", **kwargs),
    "coda_peek_page": lambda **kwargs: execute_mcp_tool("coda_peek_page", **kwargs),
    "coda_duplicate_page": lambda **kwargs: execute_mcp_tool("coda_duplicate_page", **kwargs),
    "coda_rename_page": lambda **kwargs: execute_mcp_tool("coda_rename_page", **kwargs),
    "coda_resolve_link": lambda **kwargs: execute_mcp_tool("coda_resolve_link", **kwargs)
}

def save_session(session_name, messages):
    """Save chat session to file"""
    session_file = SESSIONS_DIR / f"{session_name}.pkl"
    with open(session_file, 'wb') as f:
        pickle.dump({
            'messages': messages,
            'timestamp': datetime.now(),
            'session_name': session_name
        }, f)

def load_session(session_name):
    """Load chat session from file"""
    session_file = SESSIONS_DIR / f"{session_name}.pkl"
    if session_file.exists():
        with open(session_file, 'rb') as f:
            return pickle.load(f)
    return None

def get_available_sessions():
    """Get list of available session files"""
    sessions = []
    for session_file in SESSIONS_DIR.glob("*.pkl"):
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
                sessions.append({
                    'name': session_data['session_name'],
                    'timestamp': session_data['timestamp'],
                    'message_count': len(session_data['messages'])
                })
        except:
            continue
    return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)

def initialize_model():
    """Initialize the Gemini model"""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY environment variable not set.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        return model
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None

def process_message(user_input):
    """Process user message and get AI response"""
    if not st.session_state.model:
        st.session_state.model = initialize_model()
        if not st.session_state.model:
            return "Error: Could not initialize the AI model."
    
    # Start chat session if not exists
    if not st.session_state.chat_session:
        st.session_state.chat_session = st.session_state.model.start_chat()
    
    try:
        # Send message to AI
        response = st.session_state.chat_session.send_message(user_input, tools=[coda_tools])
        
        # Process response and handle function calls
        response_text = ""
        tool_calls_made = []
        
        while True:
            if not response.candidates:
                response_text = "No response from AI."
                break

            candidate_content = response.candidates[0].content
            if not candidate_content:
                response_text = "No content in AI response."
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
                    
                    # Store tool call info for display
                    tool_calls_made.append({
                        'name': tool_name,
                        'args': tool_args
                    })
                    
                    if tool_name in available_tool_functions:
                        try:
                            tool_output = available_tool_functions[tool_name](**tool_args)
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
                            function_responses.append(genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"error": error_message}
                                )
                            ))
                
                response = st.session_state.chat_session.send_message(function_responses)
            elif text_parts_found:
                response_text = "".join(text_parts_found)
                break
            else:
                response_text = "I received a response but couldn't interpret it."
                break
        
        # Format response with tool calls if any were made
        if tool_calls_made:
            tool_info = ""
            for tool_call in tool_calls_made:
                tool_info += f"🔧 Calling {tool_call['name']} with args: {json.dumps(tool_call['args'], indent=2)}\n\n"
            
            return f"{tool_info}{response_text}"
        else:
            return response_text
    
    except Exception as e:
        return f"Error processing message: {e}"

# Streamlit UI
def main():
    st.title("📝 Coda AI Assistant with Persistent Chat")
    
    # Sidebar with configuration and session management
    with st.sidebar:
        # Configuration Section
        st.markdown("## 🔧 Configuration")
        
        # API Keys Status
        st.markdown("### API Keys Status")
        google_key_status = "✅ Google API Key configured" if os.environ.get("GOOGLE_API_KEY") else "❌ Google API Key missing"
        coda_key_status = "✅ Coda API Key configured" if (os.environ.get("CODA_API_KEY") or os.environ.get("API_KEY")) else "❌ Coda API Key missing"
        
        st.markdown(f"{google_key_status}")
        st.markdown(f"{coda_key_status}")
        st.markdown("")
        
        # MCP Server Status
        st.markdown("### MCP Server Status")
        if st.session_state.mcp_server_process and st.session_state.mcp_server_process.poll() is None:
            st.markdown("🟢 Connected")
        else:
            st.markdown("🔴 Disconnected")
            
        if st.button("🔄 Restart Server", key="restart_server"):
            stop_mcp_server()
            with st.spinner("Restarting server..."):
                if start_mcp_server():
                    st.success("Server restarted!")
                    st.rerun()
                else:
                    st.error("Failed to restart server")
        
        st.markdown("---")
        
        # Chat Sessions Section
        st.markdown("## 💬 Chat Sessions")
        
        # Current session info with short ID
        current_session_id = st.session_state.session_name[-12:] if len(st.session_state.session_name) > 12 else st.session_state.session_name
        st.markdown(f"**Current:** {current_session_id}")
        st.markdown("")
        
        # Session management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New", key="new_session"):
                # Save current session if it has messages
                if st.session_state.messages:
                    save_session(st.session_state.session_name, st.session_state.messages)
                
                # Reset session
                st.session_state.messages = []
                st.session_state.chat_session = None
                st.session_state.session_name = f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.rerun()
        
        with col2:
            if st.button("💾 Save", key="save_session") and st.session_state.messages:
                save_session(st.session_state.session_name, st.session_state.messages)
                st.success("Saved!", icon="✅")
        
        st.markdown("---")
        
        # Saved Sessions Section
        st.markdown("## 📚 Saved Sessions")
        sessions = get_available_sessions()
        
        if sessions:
            # Show sessions in a more compact format
            for session in sessions[:5]:  # Show last 5 sessions
                session_id = session['name'][-12:] if len(session['name']) > 12 else session['name']
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{session_id}**")
                    st.caption(f"{session['message_count']} msgs • {session['timestamp'].strftime('%m/%d %H:%M')}")
                
                with col2:
                    if st.button("📂", key=f"load_{session['name']}", help="Load session"):
                        # Save current session if it has messages
                        if st.session_state.messages:
                            save_session(st.session_state.session_name, st.session_state.messages)
                        
                        # Load selected session
                        session_data = load_session(session['name'])
                        if session_data:
                            st.session_state.messages = session_data['messages']
                            st.session_state.session_name = session_data['session_name']
                            st.session_state.chat_session = None  # Reset chat session
                            st.rerun()
            
            if len(sessions) > 5:
                st.caption(f"+ {len(sessions) - 5} more sessions")
        else:
            st.markdown("No saved sessions found")
        
        st.markdown("---")
        
        # Available Commands Section
        st.markdown("## 📋 Available Commands")
        
        commands = [
            ("List documents", '"Show me all my Coda documents"'),
            ("List pages", '"List pages in document X"'),
            ("Create page", '"Create a new page called Y in document X"'),
            ("Read content", '"Show me the content of page Z"'),
            ("Update content", '"Replace the content of page Z with..."'),
            ("Append content", '"Add this content to page Z..."'),
            ("Peek at page", '"Show me the first few lines of page Z"')
        ]
        
        for cmd, example in commands:
            st.markdown(f"**{cmd}:** {example}")
            st.markdown("")
        
        st.markdown("---")
        st.markdown("*📝 Coda AI Assistant with Persistent Chat*")
    
    # Main chat interface - Create scrollable chat area
    st.markdown("### 💬 Chat")
    
    # Create a container with fixed height for scrollable chat
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.messages:
            # Display all messages in reverse order (newest at bottom)
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            # Add some space at the bottom
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            # Welcome message for new sessions
            st.markdown("""
            <div class="welcome-message">
                <h3>👋 Welcome to your Coda AI Assistant!</h3>
                <p>I can help you with:</p>
                <ul>
                    <li>📋 Listing your Coda documents and pages</li>
                    <li>✏️ Creating and editing pages</li>
                    <li>📄 Reading and updating content</li>
                    <li>🔗 Managing page links and structure</li>
                </ul>
                <p><strong>What would you like to do first?</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add separator before input
    st.markdown("---")
    
    # Chat input at the bottom - this stays fixed at bottom
    if prompt := st.chat_input("Type your message here... (e.g., 'List my documents' or 'Create a new page')"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Ensure MCP server is running
        if not st.session_state.mcp_server_process:
            with st.spinner("🔄 Starting MCP server..."):
                start_mcp_server()
        
        # Show loading indicator
        with st.chat_message("assistant"):
            with st.spinner("🤔 Processing your request..."):
                response = process_message(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-save session every 10 messages
        if len(st.session_state.messages) % 10 == 0:
            save_session(st.session_state.session_name, st.session_state.messages)
            st.success("💾 Session auto-saved!", icon="✅")
        
        # Rerun to update the display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tips:** Ask me to list your documents, create pages, manage content, or help with any Coda-related tasks!")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("Please set the GOOGLE_API_KEY environment variable")
        st.stop()
    
    if not os.environ.get("CODA_API_KEY") and not os.environ.get("API_KEY"):
        st.warning("Please set either CODA_API_KEY or API_KEY environment variable for Coda access")
    
    main()
import os
import asyncio
from google.genai import types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Set your environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "your-gemini-api-key")
os.environ["API_KEY"] = os.getenv("CODA_API_KEY", "your-coda-api-key")

async def get_coda_tools():
    """Start the Coda MCP server and return ADK-compatible tools."""
    print("🚀 Launching Coda MCP server via npx...")
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "coda-mcp@latest"],
        env={"API_KEY": os.getenv("CODA_API_KEY")}
    )
    # This uses the async factory method under the hood
    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
    print("✅ Retrieved MCP tools:", [t.name for t in tools])
    return tools, exit_stack

async def coda_agent_main():
    tools, exit_stack = await get_coda_tools()
    root_agent = Agent(
        model="gemini-1.5-flash",
        name="coda_assistant",
        instruction="""
        You are a helpful assistant for managing Coda documents and pages.
        You can list documents, show pages, create new pages with markdown, update or duplicate pages.
        Always confirm actions and ask clarifying questions if needed.
        """,
        tools=tools,
    )

    session_service = InMemorySessionService()
    session = session_service.create_session(state={}, app_name="coda_app", user_id="user_coda")
    runner = Runner(app_name="coda_app", agent=root_agent, session_service=session_service)

    prompt = "Can you list my Coda documents and then show me the pages in one of them?"
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run_async(session_id=session.id, user_id=session.user_id, new_message=content)

    async for ev in events:
        part = ev.content.parts[0]
        if part.function_call:
            print("📞 Function call:", part.function_call.name, part.function_call.args)
        elif part.function_response:
            print("📬 Function response:", part.function_response.response)
        elif part.text:
            print("💬 Text:", part.text)

    # Cleanup the MCP process
    print("🧹 Cleaning up MCP tools...")
    await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(coda_agent_main())

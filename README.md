
# 📝 Coda AI Assistant with Persistent Chat

This is a Streamlit-based AI assistant that uses the Gemini model to interact with your Coda.io documents. It leverages the `coda-mcp` (Multi-Purpose CLI) tool to perform actions like listing documents, creating and editing pages, and managing content directly from a chat interface. The application includes features for managing chat sessions, so you can save and resume your conversations.

## ✨ Features

- **Conversational Interface:** Interact with your Coda documents using natural language.
- **Persistent Sessions:** Save, load, and manage multiple chat sessions to keep track of your work.
- **Coda Actions:**
  - List all your Coda documents.
  - List pages within a specific document.
  - Create new pages.
  - Read, replace, or append content to pages.
  - Peek at the first few lines of a page.
  - Duplicate and rename pages.
  - Resolve Coda links to get metadata.
- **Real-time Status:** Monitor the status of your API keys and the `coda-mcp` server directly from the sidebar.
- **Docker Support:** Easily containerize the application for consistent deployments.

## 🚀 Getting Started

### Prerequisites

Before you begin, you'll need the following installed:

- **Python 3.8+**
- **Node.js and npm** (to run the `coda-mcp` server)
- **Coda API Key:** Get your key from [Coda API Settings](https://coda.io/account/developer).
- **Google AI Studio API Key:** Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1. Set Up Your Environment Variables

You need to provide your API keys as environment variables. Create a `.env` file in the root directory of the project with the following content:

```ini
GOOGLE_API_KEY="your_google_api_key_here"
CODA_API_KEY="your_coda_api_key_here"
````

> **Note:** The application also checks for `API_KEY` for the Coda key, but `CODA_API_KEY` is preferred for clarity.

### 2\. Installation

Clone this repository and install the required Python packages.

```bash
git clone [https://github.com/your-username/coda-ai-assistant.git](https://github.com/your-username/coda-ai-assistant.git)
cd coda-ai-assistant
pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```
streamlit
google-generativeai
python-dotenv
```

### 3\. Run the Application

Start the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application will open in your web browser. The `coda-mcp` server will be automatically started in the background the first time you interact with the AI assistant.

-----

## 🐳 Docker Deployment

For a production-ready and isolated environment, you can use Docker.

### 1\. Build the Docker Image

```bash
docker build -t coda-ai-assistant .
```

### 2\. Run the Docker Container

Provide your API keys as environment variables when you run the container.

```bash
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY="your_google_api_key_here" \
  -e CODA_API_KEY="your_coda_api_key_here" \
  coda-ai-assistant
```

The application will be accessible at `http://localhost:8501`.

-----

## 📋 Available Commands

You can use natural language to ask the assistant to perform various tasks. Here are some examples:

| Command Type       | Example Prompt                                            |
| ------------------ | --------------------------------------------------------- |
| **List documents** | "Show me all my Coda documents."                          |
| **List pages** | "List all the pages in my 'Project Planning' document."   |
| **Create page** | "Create a new page called 'Meeting Notes' in the 'Team Sync' document." |
| **Read content** | "Get the content of the 'Q3 Goals' page."                 |
| **Update content** | "Replace the content of 'Draft' page with '\# New Heading'." |
| **Append content** | "Add a new bullet point to the 'To-Do' page: '- Buy groceries'." |
| **Peek at page** | "Show me the first 10 lines of the 'Project Roadmap' page." |
| **Resolve link** | "What can you tell me about the Coda link https://www.google.com/search?q=https://coda.io/d/\_dOcId/page\_id/etc?" |

## 🤝 Contributing

Feel free to open issues or submit pull requests. All contributions are welcome\!

## 📜 License

This project is licensed under the MIT License.

```
```

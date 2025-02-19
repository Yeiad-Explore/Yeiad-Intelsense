# Yeiad-Intelsense - Project Overview

This repository contains a set of Jupyter Notebooks for building AI-powered agents and chatbots using LangChain, a framework for constructing AI workflows with integrated tools. Below is a detailed breakdown of each notebook's technical functionality.

## 1. **Customer_support_agent_v2.ipynb**

This notebook implements a customer support agent using AI-driven natural language processing (NLP) techniques. The core of this notebook involves processing user input, interpreting queries, and generating appropriate responses. The agent uses a modular pipeline where each part is designed to handle different aspects of a conversation.

### Key Code Components:
- **Input Handling**: Uses Python's `input()` method to simulate real-time user input. The input text is passed into a pre-trained NLP model for query interpretation.
- **Response Generation**: Utilizes LangChain and OpenAI's GPT model for generating human-like responses.
- **Modular Response Flow**: The agent supports multiple response paths based on user queries, categorized by topics (e.g., general queries, billing, technical support).
  
### Example Code:
```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Define OpenAI LLM
llm = OpenAI(api_key="your_openai_api_key")

# Create a chain for generating responses
response_chain = LLMChain(llm=llm)

# Sample input processing and response generation
user_input = "How can I reset my password?"
response = response_chain.run(user_input)
print(response)


## 2. **langgraph_agents.ipynb**

This notebook demonstrates the creation of AI agents using LangChain. The agents are designed to handle a series of tasks by utilizing different tools and APIs, allowing for the execution of complex workflows. The notebook covers how to initialize agents and link them to various tools, which enables the agent to interact dynamically with external data sources.

### Key Code Components:
- **Agent Initialization**: Utilizes LangChain’s `initialize_agent` to create an agent capable of dynamically selecting and executing tasks.
- **Tool Integration**: Connects external tools (e.g., search, database query) to be used by the agent to fetch data or perform specific actions.
- **Task Execution**: The agent chooses the appropriate tool based on the user’s input and executes the task accordingly.

### Example Code:
```python
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor

# Define available tools
tools = [
    Tool(name="Database Query", func=query_database, description="Fetch data from the database"),
    Tool(name="Search", func=search_web, description="Perform a web search")
]

# Initialize the agent with the available tools
llm = OpenAI(api_key="your_openai_api_key")
agent = initialize_agent(tools, llm, agent_type="zero-shot-react-description", verbose=True)

# Execute the agent with a specific task
result = agent.run("Find the latest customer feedback for product X")
print(result)



## 3. **langgraph_chatbot_withtools_v2.ipynb**

This notebook demonstrates the creation of a sophisticated chatbot integrated with external tools using LangChain. The chatbot can interact with tools such as search engines, databases, and APIs to fetch live data, making it capable of answering dynamic user queries. This notebook explores how to leverage LangChain’s agent framework to build a flexible, real-time chatbot.

### Key Code Components:
- **Tool Integration**: The chatbot integrates with external tools, such as DuckDuckGo search, to fetch real-time data based on user queries. It allows the chatbot to retrieve up-to-date information from various data sources.
- **State Management**: LangChain’s state management system tracks the context of ongoing conversations. This ensures that the chatbot can provide relevant and context-aware responses during interactions.
- **Dynamic Task Handling**: Depending on the user’s query, the chatbot dynamically selects the appropriate tool and executes the corresponding action, such as searching the web, retrieving database records, or fetching other API data.

### Example Code:
```python
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchResults

# Define the external tool for search functionality
search_tool = DuckDuckGoSearchResults()
tools = [Tool(name="Search Tool", func=search_tool.run, description="Search the web")]

# Initialize the agent with OpenAI LLM and the available tools
llm = OpenAI(api_key="your_openai_api_key")
agent = AgentExecutor(tools, llm, verbose=True)

# Example input to the chatbot
query = "What are the latest AI advancements?"
response = agent.run(query)
print(response)

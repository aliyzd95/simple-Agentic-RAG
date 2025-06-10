# Simple RAG Multi-Agent Project

This project is an AI system built on a multi-agent architecture using LangChain. A primary Router Agent is responsible for analyzing user queries and dispatching them to one of three specialized agents:

1.  **Gaming Store Agent:** Answers questions about products, orders, and customers by querying a PostgreSQL database.
2.  **AILearning Platform Agent:** Answers questions about courses, students, and instructors by querying a MongoDB database.
3.  **Weather Agent:** Provides real-time weather information for any city using an external API.

---

## Project Structure

The project is designed modularly to enhance readability, maintainability, and scalability.

```
/simple-Agentic-RAG/
|
|-- /agents/                  # Package containing the logic for specialized agents
|   |-- __init__.py
|   |-- postgres_agent.py   # Agent for interacting with the PostgreSQL database
|   |-- mongo_agent.py      # Agent for interacting with the MongoDB database
|   |-- weather_agent.py    # Agent for fetching weather information
|
|-- /utils/                   # Utility tools and helper functions
|   |-- __init__.py
|   |-- logging_config.py   # Central configuration for the logging system
|
|-- config.py                 # Central configuration file (model names, DB URIs, etc.)
|-- main.py                   # Main entry point and agent orchestrator
|-- main_langgraph.py         # Advanced implementation using LangGraph for orchestration
|-- requirements.txt          # List of required Python libraries
|-- README.md                 # This documentation file
|-- app.log                   # Log output file (generated automatically)
|-- .env                      # Local file for storing sensitive variables (must be created by you)
```

---

## Component Descriptions

* **`main.py`**: This is the heart of the application and its starting point. It is responsible for initializing the LLM, creating all agents, defining the main router agent, and managing the user interaction loop.
* **`main_langgraph.py`**: An advanced implementation that uses **LangGraph** to build the agent orchestrator as a stateful graph. This provides more explicit control over the flow of logic, better state management, and is more robust for complex, cyclical tasks.
* **`config.py`**: All configurable settings for the project, such as the LLM model name, database connection strings, and API keys, are centralized in this file.
* **`/agents/` Directory**: Each file within this directory is responsible for creating and defining the logic for one of the specialized agents. This separation of concerns results in much cleaner code.
* **`/utils/` Directory**: Contains helper code. Currently, it includes `logging_config.py`, which sets up a standard logging system to record events to both the console and the `app.log` file.
* **`.env`**: This file is used to store sensitive information like the Weather API key. It should not be committed to version control (e.g., Git).

---

## LangGraph Architecture

The `main_langgraph.py` script implements a more sophisticated architecture where the workflow is defined as a graph. This graph explicitly controls the sequence of operations.

---

## Setup and Execution

Follow these steps to run the project.

### 1. Prerequisites
* Python 3.8+
* Access to PostgreSQL and MongoDB databases (for full functionality testing).

### 2. Setup Steps

**A) Clone the Repository**
(If using Git)
```bash
git clone https://github.com/aliyzd95/simple-Agentic-RAG.git
cd simple-Agentic-RAG
```

**B) Create and Activate a Virtual Environment**
This is highly recommended to avoid conflicts with other projects.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**C) Install Dependencies**
All required libraries will be installed by running the following command:
```bash
pip install -r requirements.txt
```


**D) Create the `.env` File**
Create a new file named `.env` in the project root and add your API keys:
```
# .env
OPENAI_API_KEY="sk-..."
WEATHER_API_KEY="your_actual_openweathermap_api_key"
```
*Note: The `OPENAI_API_KEY` is required to run the `ChatOpenAI` models.*

### 3. Running the Application

After completing the setup, run the application with the following command:
```bash
python main.py
```
or 
```bash
python main_langgraph.py
```

The application will start and will be ready to accept your questions in the terminal.

---

## Sample Input and Output

Once the application is running, you can ask it various questions. The main agent will automatically route your query to the appropriate specialized agent.

**Example 1: Querying the Weather Agent**
```
Ask your question: Should I take a jacket if I go to Paris today?
```
The main agent's output (`verbose=True`) will show that the `WeatherAPI` tool was selected, and the final answer will be displayed.

**Example 2: Querying the Learning Platform Agent**
```
Ask your question: How many lessons does the 'Machine Learning Fundamentals' course have?
```
The main agent will select the `AILearningPlatformDB` tool and return the answer by querying the MongoDB database.

**Example 3: Querying the Gaming Store Agent**
```
Ask your question: What are the top 3 most expensive products in the store?
```
The main agent will select the `GamingStoreDB` tool and return the answer by executing an SQL query on the PostgreSQL database.

To exit the application, type `exit` and press Enter.

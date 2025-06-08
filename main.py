from langchain_community.agent_toolkits import create_sql_agent
# super_agent_original_structure.py

import os
from dotenv import load_dotenv
import requests
import json
from bson import json_util, ObjectId
from pymongo import MongoClient

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool, Tool
from langchain_community.utilities import SQLDatabase

load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


def create_postgres_agent() -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor for interacting with a PostgreSQL database
    specifically designed for managing a gaming store. The agent can handle queries related to products, orders,
    and customers within the gaming store database."""

    db_uri = "postgresql+psycopg2://postgres:leomessi3265@localhost:5432/test_database"
    db = SQLDatabase.from_uri(db_uri)


    POSTGRES_SYSTEM_MESSAGE = """
    You are a specialized PostgreSQL assistant dedicated to managing a gaming store database.

    Important instructions:
    - The database tables and columns are in English.
    - However, the **data content** (like order status, product names, user names, etc.) is stored in Persian (Farsi) language.
    - For example, instead of 'pending', the `orders.status` column might contain the Persian word 'در حال پردازش'.

    When generating SQL queries, ensure that any value compared in a WHERE clause reflects the correct Persian string stored in the database.
    Always use actual values used in the database, not English equivalents.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", POSTGRES_SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor


def create_mongo_agent() -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor for interacting with a MongoDB database
    used by an AI learning platform. The agent is capable of answering queries about courses, lessons, instructors,
    and users enrolled in the platform."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client["ai_learning_platform"]
    courses_collection = db["courses"]
    users_collection = db["users"]

    @tool
    def search_courses(query: str) -> str:
        """
        Use this tool ONLY for questions about course details like lessons, content, instructors, or descriptions.
        The input must be a valid Python command string using methods like find or aggregate on the 'courses_collection' variable.
        """
        local_vars = {"courses_collection": courses_collection, "ObjectId": ObjectId}
        try:
            result = eval(query, {"__builtins__": {}}, local_vars)
            if isinstance(result, int): return str(result)
            result_list = list(result)
            if not result_list: return "No courses found."
            return json.dumps(result_list, default=json_util.default, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Query execution error: {str(e)}"

    @tool
    def search_users(query: str) -> str:
        """
        Use this tool ONLY for questions about users, their enrollments, or their progress in courses.
        The input must be a valid Python command string using methods like find or aggregate on the 'users_collection' variable.
        """
        local_vars = {"users_collection": users_collection, "ObjectId": ObjectId}
        try:
            result = eval(query, {"__builtins__": {}}, local_vars)
            if isinstance(result, int): return str(result)
            result_list = list(result)
            if not result_list: return "No users found."
            return json.dumps(result_list, default=json_util.default, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Query execution error: {str(e)}"

    tools = [search_courses, search_users]

    SYSTEM_MESSAGE = """
    You are a highly specialized MongoDB assistant. Your only goal is to answer user questions by generating a complete, runnable Python command string to be executed by a tool.

    **--- YOUR PRIMARY DIRECTIVE ---**
    The input for your tools (`Action Input`) MUST be a Python string that starts with `users_collection.` or `courses_collection.`.
    NEVER output just a JSON object. ALWAYS output the full command.
    Correct format: `users_collection.find({{'full_name': 'کیان پارسایی'}})`
    Incorrect format: `{{'full_name': 'کیان پارسایی'}}`

    **--- AVAILABLE TOOLS and DATA ---**
    You have two tools:
    1. `search_courses`: Use for questions about courses, lessons, instructors. Queries MUST start with `courses_collection.`.
    2. `search_users`: Use for questions about users, enrollments, progress. Queries MUST start with `users_collection.`.

    **`courses` collection structure:**
    {{
      "title": "string", "instructor_name": "string", "lessons": [ {{ "title": "string", "content": "string" }} ]
    }}

    **`users` collection structure:**
    {{
      "full_name": "string", "enrollments": [ {{ "course_title": "string", "progress": {{ "percent_complete": "number" }} }} ]
    }}

    **--- EXAMPLES ---**
    Question: "مدرس دوره یادگیری ماشین کیست؟"
    Action: search_courses
    Action Input: `courses_collection.find({{'title': {{'$regex': 'یادگیری ماشین', '$options': 'i'}}}}, {{'_id': 0, 'instructor_name': 1}})`

    Question: "کیان پارسایی در چه دوره‌هایی ثبت نام کرده؟"
    Action: search_users
    Action Input: `users_collection.find({{'full_name': 'کیان پارسایی'}}, {{'_id': 0, 'enrollments.course_title': 1, 'enrollments.progress': 1}})`
    ---
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent,
                         tools=tools,
                         verbose=True,
                         handle_parsing_errors=True
                         )


def create_weather_agent() -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor that integrates with a weather API service.
    The agent is designed to provide up-to-date current weather information for a specified city.
    """

    @tool
    def get_current_weather(city: str) -> str:
        """Use this tool to get the current weather for a specific city."""
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {'q': city, 'appid': WEATHER_API_KEY, 'units': 'metric', 'lang': 'fa'}
        # ... (بقیه کد ابزار بدون تغییر)
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"آب و هوای فعلی در {city}: {data['weather'][0]['description']}، دما {data['main']['temp']}°C"

    tools = [get_current_weather]
    SYSTEM_MESSAGE = """"
                    تو یک دستیار مفید هستی که تنها وظیفه‌ات ارائه اطلاعات آب و هوای فعلی یک شهر مشخص است.
                    برای پاسخ به سوالات کاربر، حتماً باید از ابزار `get_current_weather` استفاده کنی.
                    پاسخ نهایی را به صورت کامل و روان به زبان فارسی به کاربر ارائه بده.
                    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent,
                         tools=tools,
                         verbose=True,
                         handle_parsing_errors=True
                         )


postgres_agent_executor = create_postgres_agent()
mongo_agent_executor = create_mongo_agent()
weather_agent_executor = create_weather_agent()

super_agent_tools = [
    Tool(
        name="GamingStoreDB",
        func=lambda q: postgres_agent_executor.invoke({"input": q}),
        description=(
            "This tool is designed to handle all user inquiries related to the gaming store. "
            "It provides detailed information about the store's products, including popular gaming consoles such as PlayStation, "
            "gaming accessories like mice, keyboards, and headsets, as well as customer-related data such as purchase history, "
            "account information, order statuses, shipping details, and payment records. "
            "Use this tool to answer questions about product availability, technical specifications, pricing, ongoing promotions, "
            "customer orders, returns, cancellations, and support issues within the gaming store ecosystem."
        )
    ),
    Tool(
        name="AILearningPlatformDB",
        func=lambda q: mongo_agent_executor.invoke({"input": q}),
        description=(
            "This tool specializes in providing comprehensive information about the AI learning platform. "
            "It covers course details including titles, lesson contents, instructor information, course schedules, and enrollment options. "
            "Additionally, it handles queries related to user profiles, course enrollments, progress tracking, certification statuses, "
            "and feedback or ratings. Use this tool to answer questions about course availability, curriculum content, instructor expertise, "
            "student progress, and administrative details within the AI education platform."
        )
    ),
    Tool(
        name="WeatherAPI",
        func=lambda q: weather_agent_executor.invoke({"input": q}),
        description=(
            "This tool is dedicated to providing accurate and real-time weather information for any specified city worldwide. "
            "It retrieves current weather conditions including temperature, humidity, wind speed, and descriptive weather states such as rain or sunshine. "
            "Use this tool to obtain up-to-date weather forecasts, alerts, and climate-related data to assist users in planning their activities accordingly."
        )
    ),
]

ROUTER_SYSTEM_MESSAGE = """
You are a specialized routing agent responsible for analyzing the user's input question and determining the most appropriate tool from the available set to handle the query.

Your primary role is to act as a dispatcher — do not attempt to answer the question yourself. Instead, carefully interpret the user's intent and context to select the single best-suited tool that can provide an accurate and relevant response.

Make sure to:
- Understand the domain and capabilities of each tool.
- Choose only one tool per user query.
- Forward the user's input as-is to the selected tool without modification.
- Avoid providing any direct answers or commentary yourself.

Available tools:
- GamingStoreDB: Handles inquiries about the gaming store products, customers, orders, and related data.
- AILearningPlatformDB: Handles queries regarding AI courses, lessons, instructors, user enrollments, and progress.
- WeatherAPI: Provides real-time weather information for specified locations.

Your goal is to optimize user satisfaction by effectively routing questions to the appropriate expert tool.
"""

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM_MESSAGE),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
# router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
router_agent = create_openai_tools_agent(llm=llm, tools=super_agent_tools, prompt=router_prompt)

main_agent_executor = AgentExecutor(
    agent=router_agent,
    tools=super_agent_tools,
    verbose=True,
    handle_parsing_errors=True
)


def ask_agent(question):
    print(f"user question: {question}")
    response = main_agent_executor.invoke({"input": question})
    print("Agent response:")
    print(response["output"])

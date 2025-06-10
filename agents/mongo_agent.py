import logging
import json
from bson import json_util, ObjectId
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pymongo import MongoClient
from langchain.tools import tool
import config

logger = logging.getLogger(__name__)


def create_mongo_agent(llm: ChatOpenAI) -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor for interacting with a MongoDB database
    used by an AI learning platform. The agent is capable of answering queries about courses, lessons, instructors,
    and users enrolled in the platform."""
    logger.info("Creating MongoDB agent executor...")

    try:
        client = MongoClient(config.MONGO_URI)
        db = client[config.MONGO_DB_NAME]
        courses_collection = db["courses"]
        users_collection = db["users"]
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}", exc_info=True)
        raise

    @tool
    def search_courses(query: str) -> str:
        """
        Use this tool ONLY for questions about course details like lessons, content, instructors, or descriptions.
        The input must be a valid Python command string using methods like find or aggregate on the 'courses_collection' variable.
        """

        logger.info(f"Executing courses search with query: {query}")

        local_vars = {"courses_collection": courses_collection, "ObjectId": ObjectId}
        try:
            result = eval(query, {"__builtins__": {}}, local_vars)
            if isinstance(result, int): return str(result)
            result_list = list(result)
            if not result_list:
                logger.warning(f"No courses found for query: {query}")
                return "No courses found."
            return json.dumps(result_list, default=json_util.default, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"MongoDB course query failed: {e}", exc_info=True)
            return f"Query execution error: {str(e)}"

    @tool
    def search_users(query: str) -> str:
        """
        Use this tool ONLY for questions about users, their enrollments, or their progress in courses.
        The input must be a valid Python command string using methods like find or aggregate on the 'users_collection' variable.
        """

        logger.info(f"Executing users search with query: {query}")

        local_vars = {"users_collection": users_collection, "ObjectId": ObjectId}
        try:
            result = eval(query, {"__builtins__": {}}, local_vars)
            if isinstance(result, int): return str(result)
            result_list = list(result)
            if not result_list:
                logger.warning(f"No users found for query: {query}")
                return "No users found."
            return json.dumps(result_list, default=json_util.default, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"MongoDB user query failed: {e}", exc_info=True)
            return f"Query execution error: {str(e)}"

    tools = [search_courses, search_users]

    SYSTEM_MESSAGE = """
    You are a highly specialized MongoDB assistant. Your only goal is to answer user questions by generating a complete, runnable Python command string to be executed by a tool.

    **--- YOUR PRIMARY DIRECTIVE ---**
    The input for your tools (`Action Input`) MUST be a Python string that starts with `users_collection.` or `courses_collection.`.
    NEVER output just a JSON object. ALWAYS output the full command.
    Correct format: `users_collection.find({{'full_name': 'Alice Johnson'}})`
    Incorrect format: `{{'full_name': 'Alice Johnson'}}`

    **--- AVAILABLE TOOLS and DATA ---**
    You have two tools:
    1. `search_courses`: Use for questions about courses, lessons, instructors, or descriptions. Queries MUST start with `courses_collection.`.
    2. `search_users`: Use for questions about users, their enrollments, or progress. Queries MUST start with `users_collection.`.

    **`courses` collection structure:**
    {{
      "_id": "ObjectId",
      "title": "string",
      "instructor_name": "string",
      "description": "string",
      "lessons": [ {{ "title": "string", "content": "string" }} ]
    }}

    **`users` collection structure:**
    {{
      "full_name": "string",
      "email": "string",
      "enrollments": [ {{ "course_id": "ObjectId", "progress": "number" }} ]
    }}
    Note: `enrollments.course_id` is a reference to the `_id` in the `courses` collection. To get course names for a user, you MUST use an `aggregate` query with `$lookup`.

    **--- EXAMPLES ---**
    Question: "Who is the instructor for 'Machine Learning Fundamentals'?"
    Action: search_courses
    Action Input: `courses_collection.find({{'title': 'Machine Learning Fundamentals'}}, {{'_id': 0, 'instructor_name': 1}})`

    Question: "What is the content of the third lesson in the Python course?"
    Action: search_courses
    Action Input: `courses_collection.find({{'title': {{'$regex': 'python', '$options': 'i'}}}}, {{'_id': 0, 'lessons': {{'$slice': [2, 1]}}, 'lessons.content': 1}})`

    Question: "What courses is Alice Johnson enrolled in and what is her progress?"
    Action: search_users
    Action Input: `users_collection.aggregate([{{'$match': {{'full_name': 'Alice Johnson'}}}}, {{'$unwind': '$enrollments'}}, {{'$lookup': {{'from': 'courses', 'localField': 'enrollments.course_id', 'foreignField': '_id', 'as': 'course_details'}}}}, {{'$unwind': '$course_details'}}, {{'$project': {{'_id': 0, 'course_title': '$course_details.title', 'progress': '$enrollments.progress'}}}}])`
    ---
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=True,
                                   handle_parsing_errors=True
                                   )

    logger.info("MongoDB agent executor created successfully.")
    return agent_executor

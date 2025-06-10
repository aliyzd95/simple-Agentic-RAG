from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

import config

from agents.mongo_agent import create_mongo_agent
from agents.postgres_agent import create_postgres_agent
from agents.weather_agent import create_weather_agent

import logging
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("==================================")
    logger.info("Starting new RAG agent session...")

    logger.info("Initializing agents...")
    llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)

    postgres_agent_executor = create_postgres_agent(llm)
    mongo_agent_executor = create_mongo_agent(llm)
    weather_agent_executor = create_weather_agent(llm)

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
    
    Do not try to force a tool to fit an irrelevant question.
    """

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_MESSAGE),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    router_agent = create_openai_tools_agent(llm=llm, tools=super_agent_tools, prompt=router_prompt)

    main_agent_executor = AgentExecutor(
        agent=router_agent,
        tools=super_agent_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    logger.info("Main router agent is ready!")
    while True:
        question = input("Ask your question: ... (type 'exit' to quit) : ")
        if question.lower() == 'exit':
            logger.info("Exit command received. Ending session.")
            print("Session ended. Goodbye!")
            break

        logger.info(f"Received user question: '{question}'")
        try:
            response = main_agent_executor.invoke({"input": question})
            logger.info(f"Agent generated response successfully.")

            print("\nAgent Response:")
            print(response.get("output", "No output generated."))
            print("-" * 30)
        except Exception as e:
            logger.error(f"An error occurred during agent invocation: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")
            print("-" * 30)


if __name__ == "__main__":
    main()

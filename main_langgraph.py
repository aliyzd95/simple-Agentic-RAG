import os
import logging
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

import config
from agents.mongo_agent import create_mongo_agent
from agents.postgres_agent import create_postgres_agent
from agents.weather_agent import create_weather_agent
from utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    input: str
    agent_outcome: str
    next_agent: Literal["PostgresAgent", "MongoAgent", "WeatherAgent", "end"]


logger.info("Initializing specialist agents...")
llm = ChatOpenAI(model=config.MODEL_NAME, temperature=config.TEMPERATURE)
postgres_agent_executor = create_postgres_agent(llm)
mongo_agent_executor = create_mongo_agent(llm)
weather_agent_executor = create_weather_agent(llm)

agent_executors = {
    "GamingStoreDB": postgres_agent_executor,
    "AILearningPlatformDB": mongo_agent_executor,
    "WeatherAPI": weather_agent_executor,
}
logger.info("Specialist agents and tools are ready.")


def router_node(state: GraphState):
    logger.info("---ROUTER NODE---")

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
    If the question is not relevant to any of these tools, you must output 'end'.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system",ROUTER_SYSTEM_MESSAGE),
        ("user", "Question: {input}")
    ])
    router_chain = prompt | llm | StrOutputParser()
    next_agent_name = router_chain.invoke({"input": state["input"]})
    logger.info(f"Router decided the next agent is: '{next_agent_name}'")
    return {"next_agent": next_agent_name}

def call_tool_node(state: GraphState):
    logger.info(f"---CALLING AGENT: {state['next_agent']}---")
    agent_name = state["next_agent"]
    selected_agent_executor = agent_executors.get(agent_name)
    if selected_agent_executor is None:
        error_message = f"Error: Agent '{agent_name}' not found."
        logger.error(error_message)
        return {"agent_outcome": error_message}
    response = selected_agent_executor.invoke({"input": state["input"]})
    agent_output = response.get("output", "Error: No output from agent.")
    logger.info(f"Agent '{agent_name}' produced output.")
    return {"agent_outcome": agent_output}


def where_to_go(state: GraphState):
    next_agent_name = state.get("next_agent", "end")
    logger.info(f"---CONDITIONAL EDGE: Deciding based on '{next_agent_name}'---")
    if next_agent_name == "end":
        logger.info("Decision: End of graph.")
        return END
    else:
        logger.info(f"Decision: Route to call_tool_node for agent '{next_agent_name}'.")
        return "call_tool_node"

workflow = StateGraph(GraphState)

workflow.add_node("router", router_node)
workflow.add_node("call_tool_node", call_tool_node)
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    where_to_go,
    {
        "call_tool_node": "call_tool_node",
        END: END
    }
)

workflow.add_edge("call_tool_node", END)
app = workflow.compile()
logger.info("LangGraph workflow has been compiled successfully.")


if __name__ == "__main__":
    logger.info("==================================")
    logger.info("Starting new LangGraph agent session...")

    while True:
        question = input("\nAsk your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            logger.info("Exit command received. Ending session.")
            print("Session ended. Goodbye!")
            break

        print("\n--- ASCII Representation of the Graph ---")
        app.get_graph().print_ascii()
        print("-----------------------------------------\n")


        logger.info(f"Received user question: '{question}'")
        try:
            final_state = app.invoke({"input": question})
            print("\n" + "="*30 + "\nAgent Response:\n" + "="*30)
            print(final_state.get("agent_outcome", "No final answer was generated."))
            print("="*30)
        except Exception as e:
            logger.error(f"An error occurred during graph invocation: {e}", exc_info=True)
            print(f"\nAn error occurred: {e}")

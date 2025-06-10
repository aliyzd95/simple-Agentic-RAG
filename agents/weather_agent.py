import logging
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
import requests
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import config

logger = logging.getLogger(__name__)


def create_weather_agent(llm: ChatOpenAI) -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor that integrates with a weather API service.
    The agent is designed to provide up-to-date current weather information for a specified city.
    """

    logger.info("Creating Weather agent executor...")

    @tool
    def get_current_weather(city: str) -> str:
        """Use this tool to get the current weather for a specific city."""
        logger.info(f"Fetching weather for city: {city}")
        params = {'q': city, 'appid': config.WEATHER_API_KEY, 'units': 'metric', 'lang': 'fa'}

        try:
            response = requests.get(config.WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return f"Current weather in {city}: {data['weather'][0]['description']}, Temperature: {data['main']['temp']}°C"

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                logger.warning(f"City not found: {city}")
                return f"Unfortunately, the city {city} is not available."
            else:
                logger.error(f"HTTP error fetching weather for {city}: {http_err}", exc_info=True)
                return f"An HTTP error occurred: {http_err}"
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching weather for {city}: {e}", exc_info=True)
            return f"An unexpected error occurred: {e}"

    tools = [get_current_weather]
    SYSTEM_MESSAGE = """
                    You are a helpful assistant whose only job is to provide the current weather for a specific city.
                    You MUST use the `get_current_weather` tool to answer the user's questions.
                    Provide the final answer to the user in a clear and complete sentence.
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

    logger.info("Weather agent executor created successfully.")
    return agent_executor

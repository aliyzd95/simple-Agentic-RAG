import logging
from langchain.agents import AgentExecutor, AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import config

logger = logging.getLogger(__name__)


def create_postgres_agent(llm: ChatOpenAI) -> AgentExecutor:
    """This function creates and returns a specialized AgentExecutor for interacting with a PostgreSQL database
    specifically designed for managing a gaming store. The agent can handle queries related to products, orders,
    and customers within the gaming store database."""

    logger.info("Creating PostgreSQL agent executor...")
    try:
        db = SQLDatabase.from_uri(config.POSTGRES_URI)

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

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )

        logger.info("PostgreSQL agent executor created successfully.")
        return agent_executor

    except Exception as e:
        logger.exception(f"Error creating PostgreSQL agent: {e}", exc_info=True)
    raise

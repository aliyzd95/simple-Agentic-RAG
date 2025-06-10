import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
POSTGRES_URI = "postgresql+psycopg2://postgres:leomessi3265@localhost:5432/game_store"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "learning_platform"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

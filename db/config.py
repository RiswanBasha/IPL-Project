import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB = os.getenv("MONGO_DB", "ipl_prediction")

    @classmethod
    def get_mongo_uri(cls):
        return cls.MONGO_URI

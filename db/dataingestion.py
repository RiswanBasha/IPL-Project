import pandas as pd
from pymongo import MongoClient
from config import Config
from functools import wraps
import logging

# Decorator for MongoDB error handling
def mongo_error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"MongoDB Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

class MongoConnector:
    def __init__(self):
        uri = Config.get_mongo_uri()
        self.client = MongoClient(uri)
        self.db = self.client[Config.MONGO_DB]

    def get_collection(self, collection_name):
        return self.db[collection_name]

class ArtifactManager:
    def __init__(self, collection_name="artifacts"):
        self.collection = MongoConnector().get_collection(collection_name)

    @mongo_error_handler
    def upload_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        records = df.to_dict("records")
        if not records:
            raise ValueError("No data to upload.")
        result = self.collection.insert_many(records)
        print(f"Inserted {len(result.inserted_ids)} records into '{self.collection.name}' collection.")
        return result.inserted_ids

    @mongo_error_handler
    def fetch_all(self, as_df=True):
        docs = list(self.collection.find())
        if not docs:
            raise ValueError("No documents found.")
        if as_df:
            df = pd.DataFrame(docs)
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])
            return df
        return docs

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Upload or fetch data from MongoDB.")
    parser.add_argument("--upload", metavar="CSV_PATH", help="Path to CSV file to upload.")
    parser.add_argument("--fetch", action="store_true", help="Fetch and print all data as DataFrame.")
    args = parser.parse_args()

    manager = ArtifactManager(collection_name="matches")

    if args.upload:
        csv_path = args.upload
        if not os.path.exists(csv_path):
            print(f"File '{csv_path}' not found.")
        else:
            manager.upload_csv(csv_path)
    if args.fetch:
        df = manager.fetch_all()
        print(df.head())

# settings.py
import os

from dotenv import load_dotenv

load_dotenv("../.env")

LOGIN = os.environ.get("LOGIN")
PASSWORD = os.environ.get("PASSWORD")
SERVER = os.environ.get("SERVER")

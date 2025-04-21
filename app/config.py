
import os
from configparser import ConfigParser

if os.path.exists('config.ini'):
    config = ConfigParser()
    config.read("config.ini", encoding='utf-8')

    DATA_STORE = config.get("GENERAL", "DATA_STORE")
    GOOGLE_CLIENT_EMAIL = config.get("GENERAL", "GOOGLE_CLIENT_EMAIL")
    GOOGLE_PRIVATE_KEY = config.get("GENERAL", "GOOGLE_PRIVATE_KEY").replace('\\n', '\n')
    AUDIENCE = config.get("GENERAL", "AUDIENCE")
    ID_DRIVE_KB = config.get("GENERAL", "ID_DRIVE_KB")
    GEMINI_API_KEY = config.get("GENERAL", "GEMINI_API_KEY")
    JIRA_AUTH_HEADER = config.get("GENERAL", "JIRA_AUTH_HEADER")  

else:
    DATA_STORE = os.getenv("DATA_STORE")
    GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
    GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n')
    AUDIENCE = os.getenv("AUDIENCE")
    ID_DRIVE_KB = os.getenv("GOOGLE_CLIENT_EMAIL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    JIRA_AUTH_HEADER = os.getenv("JIRA_AUTH_HEADER")
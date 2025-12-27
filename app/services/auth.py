import streamlit as st
from requests_oauthlib import OAuth2Session

CLIENT_ID = st.secrets["oauth"]["client_id"]
CLIENT_SECRET = st.secrets["oauth"]["client_secret"]
REDIRECT_URI = st.secrets["oauth"]["redirect_uri"]

AUTHORIZATION_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = ["openid", "email", "profile"]

def get_authorization_url():
    oauth = OAuth2Session(
        CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )

    auth_url, state = oauth.authorization_url(
        AUTHORIZATION_BASE_URL,
        access_type="offline",
        prompt="select_account"
    )

    return auth_url

def fetch_token(code):
    oauth = OAuth2Session(
        CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )
    token = oauth.fetch_token(
        TOKEN_URL,
        code=code,
        client_secret=CLIENT_SECRET
    )
    return token

def get_user_info(token):
    oauth = OAuth2Session(CLIENT_ID, token=token)
    resp = oauth.get("https://www.googleapis.com/oauth2/v2/userinfo")

    return resp.json()

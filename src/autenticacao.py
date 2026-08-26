import os
from dotenv import load_dotenv, set_key
from langchain_githubcopilot_chat import get_copilot_token

CAMINHO_ENV = ".env"

def obter_token() -> str:
    load_dotenv(CAMINHO_ENV)
    copilot_token = os.getenv("COPILOT_TOKEN")

    if not copilot_token or not copilot_token.startswith("tid="):
        print("Iniciando autenticação do GitHub Copilot...")
        copilot_token = get_copilot_token()
        if copilot_token:
            set_key(CAMINHO_ENV, "COPILOT_TOKEN", copilot_token)

    if not copilot_token:
        raise RuntimeError("Não foi possível obter um token do GitHub Copilot.")

    return copilot_token
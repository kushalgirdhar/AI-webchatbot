import sys
from app.rag.terminal_chatbot import run_chatbot

if __name__ == "__main__":
    url_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_chatbot(url=url_arg)


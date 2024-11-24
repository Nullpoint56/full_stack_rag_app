import logging
import os

import box
import yaml
from llama_index.agent.openai import OpenAIAgent

from backend.app.utils.setuppers import setup_agent

# Load configuration
environment = os.getenv("ENVIRONMENT", "dev")  # Default to 'development' if not set
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

os.environ["OPENAI_API_KEY"] = cfg.OPENAI_API_KEY

# Initialize logger
logger = logging.getLogger("uvicorn")
try:
    agent = setup_agent(cfg, logger)  # Initialize shared agent resource
except Exception as e:
    raise RuntimeError(f"Failed to initialize Agent: {e}")

# Dependency to provide the agent
def get_agent() -> OpenAIAgent:
    return agent

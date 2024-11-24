# main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

from app.utils.dependencies import logger, environment, get_agent
from app.api.routers.chat import chat_router

# Load environment variables
load_dotenv()

app = FastAPI()

# Setup CORS middleware in development mode
if environment == "dev":
    logger.warning("Running in development mode - allowing CORS for all origins")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include the chat router and inject the `get_agent` dependency
app.include_router(chat_router, prefix="/api/chat", dependencies=[Depends(get_agent)])

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", reload=True)

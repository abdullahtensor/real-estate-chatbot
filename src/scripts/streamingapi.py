"""
Real Estate Chatbot Module

This module contains a FastAPI application for a real estate chatbot powered by Zameen.com data.
"""

# python3 -m src.scripts.fastapi
import os
import asyncio
import uvicorn
from src.rag.functions import rag
from fastapi import FastAPI
from fastapi.responses import StreamingResponse


def confirguration():
    """
    Configure the project settings.

    Returns:
        bot: An instance of the bot.
    """
    # specify paths
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_db = os.path.join(project_folder, "data", "vector-db")

    # load model
    # error, because rag function was updated with singleton
    bot, streamer = rag(path_to_db=path_db, use_api=True)
    del streamer
    return bot


app = FastAPI()
chatbot = confirguration()


@app.get("/")
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Zameen.com Chatbot API!"}


# function that mimics how streaming response might work, replace it with streamer recieved from local llm
async def split_and_stream(input_text):
    words = input_text.split()
    await asyncio.sleep(3)
    for word in words:
        await asyncio.sleep(0.5)
        yield word


@app.get("/stream")
async def process_prompt(query: str):
    """
    Endpoint to process user prompts and generate chatbot responses.

    Parameters:
        - prompt (str): The user's prompt for the chatbot.

    Returns:
        dict: A dictionary containing the chatbot response.
    """
    response = chatbot.run({"query": query})
    return StreamingResponse(split_and_stream(response), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
FastAPI Chatbot Application

This module defines a FastAPI application for following tasks:
- uploading
- processing
- chatting
"""

import os
from typing import Dict, List
from fastapi import UploadFile, Form, APIRouter
from rag.functions import (
    rag,
    loadnsplit_data,
    create_vector_db,
    contruct_prompt,
    update_history,
    initial_questions,
)
from utils import get_logger
from application_context import resources

logger = get_logger("chat")

router = APIRouter(
    prefix="/chatdoc",
    tags=["chatbot"],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)

bot, uploaded, splits = None, False, None
chat_history: Dict[str, List[str]] = {}

path_project = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
public = os.path.join(path_project, "public")


def configuration(file, name):
    """
    Configure the project settings and create necessary files and databases.

    Parameters:
    - file: Upload file object containing data to be processed.

    Returns:
    - bot: Instance of a bot object.
    - streamer: Instance of a streamer object.
    """
    global bot, uploaded, splits  # [global-statement]

    # specify paths
    path_data = os.path.join(path_project, "data", "uploaded", file.filename)
    use_openai_api = os.environ.get("USE_OPENAI_API", "FALSE")
    if use_openai_api == "TRUE":
        path_db = os.path.join(path_project, "data", "vector-db-openai")
    else:
        path_db = os.path.join(path_project, "data", "vector-db-hf")

    # save uploaded file
    with open(path_data, "wb") as f:
        f.write(file.file.read())

    # create vector db
    all_splits = loadnsplit_data(path_data)
    create_vector_db(resources, all_splits, direct=path_db)

    # initialize agent
    bot, streamer = rag(resources, path_to_db=path_db)
    del streamer
    uploaded = True

    os.remove(path_data)

    # initialize chat history
    if name not in chat_history:
        chat_history[name] = []
        splits = all_splits


@router.post("/generatequestions")
def generate_questions():
    questions = initial_questions(splits, resources)
    return {"questions": questions}


@router.post("/upload")
async def uploadfile(files: list[UploadFile], name: str = Form(...)):
    """
    Process and save a list of uploaded files.

    Parameters:
    - files (List[UploadFile]): List of objects representing the files to be processed.

    Returns:
    - dict: A dictionary containing a message indicating result of the saving process.
    """
    try:
        for file in files:
            configuration(file, name)
            return {"message": "Document Uploaded. Proceed Towards Chatbot!"}
    except Exception as e:
        return {"message": f"ERROR : Unable To Process Document. {e}"}


@router.get("/check-upload-status")
async def check_upload_status():
    try:
        if uploaded:
            return {
                "status": "processed",
                "message": "Document is processed. Proceed to Chatbot!",
            }
        else:
            return {
                "status": "pending",
                "message": "Document is still being processed. Please wait.",
            }
    except Exception as e:
        return {"status": "error", "message": f"Error: {e}"}


@router.post("/chatbot")
def process_prompt(prompt: str = Form(...), name: str = Form(...)):
    """
    Endpoint to process user prompts and generate chatbot responses.

    Parameters:
        - prompt (str): The user's prompt for the chatbot.

    Returns:
        dict: A dictionary containing the chatbot response.
    """
    global chat_history
    try:
        if (uploaded is True) and (bot is not None):
            template = contruct_prompt(chat_history, name, prompt)
            response = bot.run({"query": template})
            chat_history = update_history(chat_history, name, prompt, response)
        else:
            response = "Error : Document Required. Unable To Chat"
    except Exception as e:
        response = f"Error : Something went wrong. Recheck user name. {e}"
    return {"Response": response}

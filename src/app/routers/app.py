"""Chatbot page for the application
this is a non api HTML reponse router
"""

import os
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi import Request
from app.dependencies import templates

router = APIRouter(
    prefix="/chatdoc",
    tags=["chatbot"],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)


@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "chatbot.html", {"request": request, "URL": os.environ.get("URL")}
    )


# @router.get("/", response_class=HTMLResponse)
# async def read_root():
#     """
#     Endpoint to serve the HTML page for document upload and chatbot interaction.
#     """
#     path_html = os.path.join(path_project, "views", "chatbot.html")
#     return FileResponse(path_html)

"""The main fastapi app module

Returns:
    None
"""

import os
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from utils import get_logger
from app.routers import index
from app.routers import app as chatapp
from app.routers import realestate as realestateapp

from app.routers.api import user
from app.routers.api import index as api_index
from app.routers.api import app as chatbot
from app.routers.api import realestate as realestatebot

from dotenv import load_dotenv

load_dotenv()
# load_dotenv(get_full_path("../.env"))

logger = get_logger("main")


description = """
Centrox AI template. Fire way!! 🚀

## APIs and usecases implemented on this server:
- usecase 1
- usecase 2

"""

app = FastAPI(
    title="Centrox AI Template",
    description=description,
    summary="Modify this to wite your own summary",
    version="0.0.1",  # run to get current version: semantic-release version --print
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Muhammad Harris",
        "url": "https://www.centrox.ai",
        "email": "wizard@centrox.ai",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redocs",
)


def apply_origins(application: FastAPI):
    origins: str | None = os.getenv("ORIGINS")
    if origins is None:
        origins = "*"

    origins_lst = origins.split(",")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return origins


origins_applied = apply_origins(app)

app.mount("/public", StaticFiles(directory="../public"), name="public")

logger.info(f"Accepting from origins {origins_applied}")
app.include_router(index.router)
# 1
app.include_router(chatapp.router)
app.include_router(realestateapp.router)

api_v1_router = APIRouter(
    prefix="/api/v1",
    tags=[],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)

# 2
api_v2_router = APIRouter(
    prefix="/api/v2",
    tags=[],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)

"""
Include All of the application api routers to this 
router object
"""
api_v1_router.include_router(api_index.router)
api_v1_router.include_router(user.router)
# 3
api_v2_router.include_router(chatbot.router)
api_v2_router.include_router(realestatebot.router)

app.include_router(api_v1_router)
# 4
app.include_router(api_v2_router)

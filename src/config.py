"""All Configureables should go here

Returns:
    None
"""

import os
from utils import get_full_path
from dotenv import load_dotenv


class Printable:
    def _get_attributes(self):
        attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        ]

        return attributes

    def _get_dict(self):
        attributes = self._get_attributes()
        d = {i: getattr(self, i) for i in attributes}
        return d

    def __repr__(self):
        d = self._get_dict()
        s = str(d)
        return s

    def __iter__(self):
        for k, v in self._get_dict().items():
            yield k, v


class Config(Printable):
    DB = "centrox"
    COLLECTION = "cloths"
    AWS_ACCESS_KEY_ID = "AKIAV44TK2KFG2IRHL7V"
    AWS_SECRET_ACCESS_KEY = "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"
    INDEX_BUCKET = "imagesearch-indexes"
    MONGO_HOST = (
        "a8fe1206c092511eab2b60a1835f0bf3-2138289406.ap-south-1.elb.amazonaws.com"
    )
    MONGO_USERNAME = "admin"
    MONGO_PASSWORD = "Qwerty12"
    MONGO_AUTHSOURCE = "centrox"
    MONGO_DATABASE = "centrox"
    REDIS_HOST = "174.138.121.187"
    REDIS_PASSWORD = "Qwerty12#$"
    REDIS_PORT = "6379"
    AI_URL = "https://ai.centrox.xyz/api/predict/imagevec"
    CLOTH_COUNT_URI = "https://www.centrox.xyz/api/cloth/getcount"
    GENDER_HOST = "139.59.54.148"
    IMAGE_RETRIEVAL_HOST = "139.59.49.109"
    IMAGE_RETRIEVAL_MODELNAME = "resnet_encoder_inter"
    IMAGE_RETRIEVAL_VERSION = 1
    IMAGE_SIZE = 128
    EMBEDDING_SIZE = 128
    CLOSEST_TOP_K = 10
    BATCH_SIZE = 50
    INDEX_DIR = get_full_path("../data")


class Ragconfig:
    """
    Configuration class for RAG-related parameters.

    This class defines constants and configuration parameters used in the RAG
    (Retrieval-Augmented Generation) pipeline.
    """

    load_dotenv()

    TOGETHERAI_API_KEY = os.environ.get("TOGETHERAI_API_KEY", None)
    if TOGETHERAI_API_KEY is None:
        raise ValueError(
            "TOGETHERAI_API_KEY environment variable is not set. Please set a value."
        )

    use_openai = os.environ.get("USE_OPENAI_API", "FALSE")
    if use_openai == "TRUE":
        openai_apikey = os.getenv("OPENAI_API_KEY", None)
        if openai_apikey in ["", None]:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set a value. Or set USE_OPENAI_API = FALSE."
            )

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    RETRIEVED_CHUNKS = int(os.getenv("RETRIEVED_CHUNKS", "4"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "192"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
    EMBEDDING_NAME = os.getenv("EMBEDDING_NAME", "WhereIsAI/UAE-Large-V1")
    HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "NousResearch/Llama-2-7b-chat-hf")
    API_MODEL_NAME = os.getenv("API_MODEL_NAME", "togethercomputer/llama-2-7b-chat")

    API_MODEL_NAME_2 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    HF_MODEL_NAME_2 = "TheBloke/Llama-2-7B-GPTQ"

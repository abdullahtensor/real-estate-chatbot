"""
This module provides a SingleInstance class that ensures only 
one instance is created and initializes necessary resources.
"""

import os
import logging
from rag.functions import together_ai, llm_pipeline, embeddings_model, openai_embeddings

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceInitializer:
    """
    This class implements the Singleton pattern to ensure only one instance
    is created. It initializes resources such as embeddings and LLM.

    Attributes:
        _instance (SingleInstance): The single instance of the class.

    Methods:
        __new__: Creates a new instance or returns the existing one.
        initialize_resources: Initializes resources like embeddings and LLM.
    """

    _instance = None

    def __new__(cls):
        """Create a new instance or return the existing one."""

        if not cls._instance:
            cls._instance = super(ResourceInitializer, cls).__new__(cls)
            cls._instance.initialize_resources()
        return cls._instance

    def initialize_resources(self):
        """Initialize resources like embeddings and LLM."""

        try:
            use_openai_api = os.environ.get("USE_OPENAI_API", "FALSE")
            if use_openai_api == "TRUE":
                self.embeddings = openai_embeddings()
                logger.info("\nOPENAI Embeddings model initialized successfully")
            elif use_openai_api == "FALSE":
                self.embeddings = embeddings_model()
                logger.info("\nHuggingFace Embeddings model initialized successfully")

            use_api = os.environ.get("USE_API", "TRUE")
            if use_api == "TRUE":
                self.model = together_ai()
                logger.info("\nAPI model (together_ai) initialized successfully")
            elif use_api == "FALSE":
                self.model, self.streamer = llm_pipeline()
                logger.info("LLM pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Initialization of resources failed: {str(e)}")


resources = ResourceInitializer()

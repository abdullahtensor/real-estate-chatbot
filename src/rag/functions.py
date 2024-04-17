"""
Module for building a LLM pipeline and using it in a Retrieval-Based Generative model.

This module imports various libraries, modules and utilities for working with LLMs.
It provides functions for setting up different components such as models & pipelines.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_together import Together

import time
import random
import torch
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM, pipeline  # type: ignore

from config import Ragconfig as ragConfig

#########################################################
#### Document loader, splitter & embeddings database ####
#########################################################


def loadnsplit_data(file_path):
    """
    Load and split data from a PDF or CSV file.

    Parameters:
    - file_path (str): The path to the input file.

    Returns:
    - all_splits (list): A list of text splits obtained from the loaded documents.
                        Each element in the list represents a chunk of text.
    """

    # format based processing
    if file_path.lower().endswith(".pdf"):
        # load file
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()

        # split to chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ragConfig.CHUNK_SIZE, chunk_overlap=ragConfig.CHUNK_OVERLAP
        )
        all_splits = text_splitter.split_documents(documents)

    elif file_path.lower().endswith(".csv"):
        # load file
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()

        # no need of splitting
        all_splits = documents
    else:
        return None

    return all_splits


def embeddings_model():
    """
    Selects the appropriate device (CPU or CUDA) and loads pre-trained
    embeddings using Hugging Face's transformers library.

    Returns:
        HuggingFaceEmbeddings: An instance of the pre-trained embeddings model.
    """
    # select device type
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Pre-Trained Embeddings
    embedds = HuggingFaceEmbeddings(
        model_name=ragConfig.EMBEDDING_NAME, model_kwargs={"device": device}
    )
    return embedds


def openai_embeddings():
    """
    Fetches embeddings from OpenAI's text-embedding-3-small model.

    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings with the specified model.
    """
    embedds = OpenAIEmbeddings(model="text-embedding-3-small")
    return embedds


def create_vector_db(singleton, splits, direct):
    """
    Create a vector database (Chroma) for retrieval-based models.

    Parameters:
    - splits (list): List of text splits/documents to be stored in the vector database.
    - device (str): The device (CPU or CUDA) to use for the embeddings.
    - direct (str): The directory to persist the vector database.

    Returns:
    - retriever (ChromaRetriever): The retriever created from the vector database.
    """
    # Load Pre-Trained Embeddings
    embedds = singleton.embeddings

    # Create Database
    Chroma.from_documents(documents=splits, embedding=embedds, persist_directory=direct)


def load_vector_db(singleton, direct):
    """
    Load a vector database with specified settings.

    Args:
        direct (str): The directory to persist the database.
        k (int): The number of neighbors to retrieve (default is 4).

    Returns:
        retriever: A retriever instance for the loaded vector database.
    """

    # Load Pre-Trained Embeddings
    embedds = singleton.embeddings

    # Create Database
    vectordb = Chroma(persist_directory=direct, embedding_function=embedds)
    retriever = vectordb.as_retriever(search_kwargs={"k": ragConfig.RETRIEVED_CHUNKS})

    return retriever


#########################################################
############### Prompt Templates For LLMs ###############
#########################################################


def prompt_format(instruction_prompt=None, system_prompt=None):
    """
    Generate a prompt template for the language model.

    Parameters:
    - instruction_prompt (str, optional): The instruction prompt template.
    - system_prompt (str, optional): The system prompt template. Defaults to None.

    Returns:
    - template (str): The assembled prompt template containing both prompts.
    """

    ## Tags (Instructions & System)
    begin_instruction, end_instruction = "[INST]", "[/INST]"
    begin_system, end_system = "<<SYS>>\n", "\n<</SYS>>\n\n"

    ## System Prompt
    if system_prompt is None:
        system_prompt = """\
        You are a helpful, respectful and honest assistant that answers user's queries by looking at context. Give short and precise answers. Your answer should not exceed 180 tokens.
        Always answer helpfully and to the point using the context text only. Your answers should only answer the question once and not have any text after the answer is done.
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 
        """

    ## User Prompt
    if instruction_prompt is None:
        instruction_prompt = """
        Answer the questions by looking at context.
        <ctx> 
            Context:{context} 
        </ctx>
        {question}      """

    ## Assembled Prompt (System & User)
    system = begin_system + system_prompt + end_system
    template = begin_instruction + system + instruction_prompt + end_instruction
    return template


def prompt_template():
    """
    Create a prompt template and initialize a memory buffer for conversation history.

    Returns:
    - prompt (PromptTemplate): The initialized PromptTemplate object.
    """

    ## Prompt Format
    template = prompt_format()
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    ## Memory Buffer
    # memory = ConversationBufferMemory(memory_key="history", input_key="question")

    return prompt


#########################################################
############### API, GGUF & Quantized LLM ###############
#########################################################


def together_ai():
    """
    Create and initialize a Together AI language model.

    Parameters:
    - model_name (str, optional): The name of the Together AI language model.
    - temperature (float, optional): The parameter for randomness in text generation.
    - tokens (int, optional): The maximum number of tokens to generate.

    Returns:
    - llm (Together): The initialized Together AI language model.
    """

    api_key = ragConfig.TOGETHERAI_API_KEY

    llm = Together(
        model=ragConfig.API_MODEL_NAME,
        temperature=ragConfig.TEMPERATURE,
        max_tokens=ragConfig.MAX_TOKENS,
        together_api_key=api_key,
    )

    return llm


def llm_pipeline():
    """
    Load a llm pipeline for text generation using Hugging Face's Transformers.

    Parameters:
    - name (str, optional): The name or path of the Hugging Face model to be loaded.

    Returns:
    - llm (HuggingFacePipeline): The initialized pipeline for text generation.
    """

    # Load Tokenizer
    mname = ragConfig.HF_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(mname)

    # Streamer
    streamer = None
    # streamer = TextIteratorStreamer(
    #    tokenizer, skip_prompt=True, skip_special_tokens=True
    # )

    # Load Pre-Trained Model
    model = AutoModelForCausalLM.from_pretrained(
        mname, device_map="auto", torch_dtype=torch.float16, load_in_4bit=True
    )

    # Load Transformer Pipeline
    transformer_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=ragConfig.MAX_TOKENS,
        do_sample=True,
        top_k=30,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        # streamer=streamer,
    )

    # Load Hugging-Face Pipeline via LangChain
    llm = HuggingFacePipeline(pipeline=transformer_pipeline)

    return llm, streamer


#########################################################
############# Retrieval-Augmented Generation ############
#########################################################


def rag(singleton, path_to_db):
    """
    Create and initialize a Retrieval-Based Generative (RAG) model.

    Parameters:
    - uploaded_files (UploadFile, optional): The uploaded file object. Defaults to None.

    Returns:
    - agent (RetrievalQA): The initialized RetrievalQA agent.
    """

    retriever = load_vector_db(singleton, direct=path_to_db)

    ## Load LLM Pipeline
    streamer = None
    llm = singleton.model

    ## Initialize Llama Prompt Template
    prompt = prompt_template()
    chain_type_kwargs = {"prompt": prompt}

    ## Initialize Retrieval Based Generative Model
    agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        chain_type_kwargs=chain_type_kwargs,
    )

    return agent, streamer


#########################################################
############ Multi-User Chat & Prompt Setting ###########
#########################################################


def contruct_prompt(history, name, prompt):
    """
    Construct a conversation prompt by combining history and the current prompt.

    Parameters:
        name (str): The name or identifier for the chat history.
        prompt (str): The human's prompt in the current conversation.

    Returns:
        str: The constructed conversation prompt.
    """
    chat = "\n".join(history[name])
    message = f"""
    <hs> 
        History:
        {chat}
    </hs> 
    Question: {prompt}
    Response: """  # not sure

    return message


def update_history(history, name, prompt, response):
    """
    This function updates the chat history for a given 'name' with the provided
    'prompt' and 'response'. It ensures that the chat history does not exceed a
    certain length, removing older entries if necessary.

    Parameters:
        name (str): The name or identifier for the chat history.
        prompt (str): The human's prompt in the conversation.
        response (str): The AI's response in the conversation.

    Returns:
        None
    """
    if len(history[name]) > 6:
        history[name].pop(0)
        history[name].pop(0)

    history[name].append("[Human]: " + prompt)
    history[name].append("[AI]:" + response)
    return history


#########################################################
############ Initial and Random LLM Questions ###########
#########################################################


def generate_indexes(n):
    """
    Generate 4 random numbers between 0 and the given number (exclusive).

    Returns:
    list: A list of 4 randomly generated numbers.
    """
    if n < 1:
        raise ValueError(
            "Vector Database is empty. Unable to generate initial questions!"
        )

    elif n == 1:
        return [0, 0, 0, 0]

    if n in [2, 3]:
        numbers = random.sample(list(range(n)) * 2, 4)
    else:
        numbers = random.sample(range(n), 4)

    return numbers


def initial_questions(all_splits, singleton):
    # Generate Random Indexes
    indexes = generate_indexes(len(all_splits))

    # Setup Prompt Template
    template = """
    <s>[INST]
        <<SYS>>You are a professional interviewer who asks only one short question based on given document : `Document`.<</SYS>>
    Follow these instructions carefully before responding.
    - Do not add question number. 
    - Do not use any jargons like `according to the document` or `according to the study` at the end of question.
    - Question should be simple, short, concise and to the point.
    - The answer of the queston should exsist in the given document.

    Document: {chunk}
    Short Question: [/INST]
    """
    prompt = PromptTemplate(input_variables=["chunk"], template=template)

    # Initialize Chain
    chain = LLMChain(llm=singleton.model, prompt=prompt)

    # Generate Questions
    questions = []
    for i in indexes:
        time.sleep(1)
        chunk = all_splits[i].page_content
        response = chain.run({"chunk", chunk})
        questions.append(response)
    return questions

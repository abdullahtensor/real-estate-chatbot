import os
import json
from langchain import hub
from langchain.tools import Tool
from langchain import PromptTemplate
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic


class PersonalDetails(BaseModel):
    name: str = Field(
        default=None,
        description="Full name of user.",
    )
    city: str = Field(
        default=None,
        description="Name of city user lives in.",
    )
    beds: int = Field(
        default=None,
        description="Number of bedrooms user wants in his house.",
    )
    baths: int = Field(
        default=None,
        description="Number of baths user wants in his house.",
    )
    price: int = Field(
        default=None,
        description="Average price of the house user is expecting.",
    )


class DetailsForm:
    def __init__(self):
        self.llm = get_llm()
        self.chain = LLMChain(llm=self.llm, prompt=self.get_prompt())
        self.extractor = create_tagging_chain_pydantic(PersonalDetails, self.llm)

    def get_prompt(self):
        prompt = ChatPromptTemplate.from_template(
            """Below is are some things to ask the user for in a coversation way.
            You should only ask one question at a time even if you don't get all the information don't ask as a list!
            Don't greet the user! Don't say Hi. If the ask_for list is empty then thank them and ask how you can help them.


            ### ask_for list: {ask_for}"""
        )
        return prompt

    def ask_details(self, required_fields):
        return self.chain.run(ask_for=required_fields)

    def add_details(
        self, current_details: PersonalDetails, new_details: PersonalDetails
    ):
        non_empty_details = {
            k: v for k, v in new_details.dict().items() if v not in [None, ""]
        }
        updated_details = current_details.copy(update=non_empty_details)
        return updated_details

    def remaining_fields(self, details):
        remaining = []
        for field, value in details.dict().items():
            # update this to resolve bed/bath int issue
            if value in [None, "", 0]:
                remaining.append(f"{field}")
        return remaining

    def update_details(self, user_input, given_details):
        extracted_details = self.extractor.run(user_input)
        given_details = self.add_details(given_details, extracted_details)
        remaining_details = self.remaining_fields(given_details)

        return given_details, remaining_details


def tool_form():
    form = DetailsForm()

    # create tool
    tool = Tool(
        name="fill_form",
        func=form.fill_form,
        description="A function that asks questions from user related to personal information.",
    )
    return tool


def load_csv_retriever(path):
    # load file & embeddings model
    loader = CSVLoader(file_path=path)
    data = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # load vector db
    db = Chroma.from_documents(data, embeddings)
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.50, "k": 3},
    )

    return retriever


def tool_inspection():
    # initialize retriever
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "react-content",
        "inspection_instructions.csv",
    )
    retriever = load_csv_retriever(path)

    # create tool
    tool = create_retriever_tool(
        retriever,
        "property_instructions",
        "Returns instructions and details about anything related to property inspection.",
    )
    return tool


def tool_recommender():
    # initialize retriever
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "react-content",
        "Properties.csv",
    )
    retriever = load_csv_retriever(path)

    # create tool
    tool = create_retriever_tool(
        retriever,
        "property_recommender",
        "Returns properties related to user query.",
    )
    return tool


def get_tools():
    # load all tools
    tool1 = tool_inspection()
    tool2 = tool_recommender()

    tools_list = [tool1, tool2]
    return tools_list


def get_agent_prompt():
    # analyze prompt
    prompt = hub.pull("hwchase17/openai-tools-agent")
    return prompt


def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613", temperature=0, streaming=False, verbose=False
    )
    return llm


def create_agent():
    # load agent resources llm, tools and prompt
    prompt = get_agent_prompt()
    llm = get_llm()
    tools = get_tools()

    # agent initializer and executor
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    return agent_executor


class IntentClassifier:
    def __init__(self):
        self.llm = get_llm()
        self.chain = LLMChain(llm=self.llm, prompt=self.get_prompt())

    def get_prompt(self):
        prompt = """You will be given a question and your job is to classify to which service does that question belongs.

      These are the available services.
      1. Personal information
      2. Other

      Response format should be:
      ["service_number": "number of the service being selected", "service_name": "name of the service being selected"]

      Question: {query}
      RFC8259 compliant JSON response: """

        prompt_template = PromptTemplate(input_variables=["query"], template=prompt)

        return prompt_template

    def json_decoder(self, response):
        # convert string to dictionary
        to_dict = response.replace("[", "{").replace("]", "}")
        data = json.loads(to_dict.replace(":", ": "))

        # extract & return
        return data["service_number"], data["service_name"]

    def classify(self, question):
        # classify & decode intent of user prompt
        response = self.chain.run(question)
        service_number, service_name = self.json_decoder(response)

        return service_number

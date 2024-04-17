from react.functions import create_agent, IntentClassifier, DetailsForm, PersonalDetails
from utils import get_logger
from fastapi import Form, APIRouter

logger = get_logger("realestate-bot")

router = APIRouter(
    prefix="/realestatebot",
    tags=["realestatebot"],
    dependencies=[],
    responses={404: {"message": "Not found", "code": 404}},
)

# global variables, same for every user
classifier = IntentClassifier()
agent = create_agent()
form = DetailsForm()

# global variables different for every user
user_details = PersonalDetails()
required_fields = list(PersonalDetails.schema()["properties"].keys())
previous = None


@router.post("/chatbot")
def response(question: str = Form(...), name: str = Form(...)):
    global user_details, required_fields, previous
    try:
        solution = None
        previous_length = len(required_fields)

        # continuation of previous service
        if previous == "1":
            # update exsisting form & ask next question
            user_details, required_fields = form.update_details(question, user_details)
            solution = form.ask_details(required_fields)

            # check if user wants to exit current service
            if previous_length == len(required_fields):
                # classify intent of user prompt
                service_number = classifier.classify(question)

                # verify if user wants to exit
                if service_number == "2":
                    solution = agent.invoke({"input": question})["output"]
                    previous = service_number
            else:
                previous_length = len(required_fields)
                # info have been recorded message
                solution = form.ask_details(required_fields)

        # utilization of new service
        else:
            # classify intent of user prompt
            service_number = classifier.classify(question)

            # service requiring conversational flow
            if service_number == "1":
                solution = form.ask_details(required_fields)
            # other services
            elif service_number == "2":
                solution = agent.invoke({"input": question})["output"]
            # error message for in appropiate option selection
            else:
                solution = service_number

            previous = service_number

        answer = solution
    except Exception as e:
        answer = f"Error: {e}"

    return {"Response": answer}

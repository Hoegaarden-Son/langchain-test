from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def chat_test():
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
        # base_url="...",
        # organization="...",
        # other params...
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    print(ai_msg)



class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


def classify_text(text: str) -> Classification:
    """
    Classify the text into sentiment, aggressiveness, and language.
    """
    llm = init_chat_model("gpt-4o", model_provider="openai")

    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )


    # Structured LLM
    structured_llm = llm.with_structured_output(Classification)


    inp = "너 비트코인 좀 알아?"
    prompt = tagging_prompt.invoke({"input": inp})
    response = structured_llm.invoke(prompt)

    print(response)
    return response

classify_text("")
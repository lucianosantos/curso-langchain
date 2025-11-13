from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

@chain
def square (input_dict: dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x * x}

question_template = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}",
)

gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

chain = square | question_template | gemini

result = chain.invoke({"x": 5})

print(result.content)
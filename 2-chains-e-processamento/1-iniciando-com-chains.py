from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

question_template = PromptTemplate(
    input_variables=["name"],
    template="Hello, I'm {name}! Tell me a joke with my name!",
)

gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

chain = question_template | gemini

# result = gemini.invoke(question_template.format(name="Alice"))
# translates into
result = chain.invoke({"name": "Alice"})

print(result.content)
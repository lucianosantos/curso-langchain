from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to english: \n{initial_text}",
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 words: \n{text}",
)

gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

translate = template_translate | gemini | StrOutputParser()
pipeline = {"text": translate} | template_summary | gemini | StrOutputParser()

result = pipeline.invoke({"initial_text": "Langchain é um framework para desenvolvimento de aplicações com IA."})

print(result)
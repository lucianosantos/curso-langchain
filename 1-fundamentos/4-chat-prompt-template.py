from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

system = ("system", "You are a helpful assistant that answers questions in a {style} style.")
user = ("user", "{question}")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="casual", question="What's the weather like today?")

for msg in messages:
    print(f"{msg.type}: {msg.content}")

gemini = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
result = gemini.invoke(messages)

print(result.content)
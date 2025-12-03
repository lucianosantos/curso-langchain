from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{history}"),
    ("user", "{input}")
])

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0.9)

chain = prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "demo-session"}}

# Interactions
response1 = conversational_chain.invoke(
    {"input": "Hello, my name is Luciano, how are you?"},
    config=config
)

print(response1)
print("-" * 30)

response2 = conversational_chain.invoke(
    {"input": "Can you repeat my name?"},
    config=config
)

print(response2)
print("-" * 30)

response3 = conversational_chain.invoke(
    {"input": "Can you repeat my name in a motivation phrase?"},
    config=config
)

print(response3)
print("-" * 30)
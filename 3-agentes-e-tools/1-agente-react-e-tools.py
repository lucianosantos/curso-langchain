from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result."""
    try:
        result = eval(expression) # Note: Using eval can be dangerous; in production, use a safe parser.
    except Exception as e:
        return f"Erro ao calcular expressão: {e}"
    
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mock web search tool that returns a fixed response."""
    data = {"Brazil": "Brasilia", "Portugal": "Lisboa", "Spain": "Madrid", "France": "Paris"}

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"A capital de {country} é {capital}."
        
    return "Desculpe, não encontrei informações relevantes."

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0, streaming=False)
tools = [calculator, web_search_mock]

agent = create_agent(llm, tools)

print(agent.invoke({"messages": [{"role": "user", "content": "How much is 10 + 10?"}]}))
print(agent.invoke({"messages": [{"role": "user", "content": "What is the capital of Iran?"}]}))
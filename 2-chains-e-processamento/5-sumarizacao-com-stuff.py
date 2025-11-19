from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

long_text = """
SONETO DE FIDELIDADE


De tudo, ao meu amor serei atento
Antes, e com tal zelo, e sempre, e tanto
Que mesmo em face do maior encanto
Dele se encante mais meu pensamento.

Quero vivê-lo em cada vão momento
E em louvor hei de espalhar meu canto
E rir meu riso e derramar meu pranto
Ao seu pesar ou seu contentamento.

E assim, quando mais tarde me procure
Quem sabe a morte, angústia de quem vive
Quem sabe a solidão, fim de quem ama

Eu possa me dizer do amor (que tive):
Que não seja imortal, posto que é chama
Mas que seja infinito enquanto dure.
"""

# CharacterTextSplitter ou RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter:
# É mais inteligente. Entende a estrutura do texto, quebras de linha, parágrafos, etc.
# Também evita cortar palavras ao meio.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    # Sobreposição entre os chunks. Pega 70 caracteres do final do chunk anterior e adiciona no próximo.
    # Isso ajuda a manter o contexto entre os chunks.
    chunk_overlap=70,
)

parts = text_splitter.create_documents([long_text])

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)

# Create a summarization prompt
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in Portuguese:\n\n{text}\n\nSummary:"
)

# Create the summarization chain
chain_summarize = summarize_prompt | llm | StrOutputParser()

# Combine all text parts into one
combined_text = "\n\n".join([doc.page_content for doc in parts])

result = chain_summarize.invoke({"text": combined_text})

print(result)
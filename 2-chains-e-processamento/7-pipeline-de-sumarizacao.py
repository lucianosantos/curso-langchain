from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
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

# CUSTOM MAP_REDUCE STRATEGY
map_prompt = PromptTemplate.from_template(
    template="Resuma o seguinte trecho de texto de forma concisa:\n\n{context}"
)
map_chain = map_prompt | llm | StrOutputParser()

prepare_map_inputs = RunnableLambda(lambda docs: [{"context": doc} for doc in docs])
map_stage = prepare_map_inputs | map_chain.map()

reduce_prompt = PromptTemplate.from_template(
    template="Combine os seguintes resumos em um único resumo coerente e completo em Português:\n{context}"
)
reduce_chain = reduce_prompt | llm | StrOutputParser()

prepare_reduce_input = RunnableLambda(lambda summaries: {"context": "\n".join(summaries)})

# Chama o map_stage passando o context como param =>
# Chama o prepare_reduce_input passando o resultado anterior como param =>
# Chama o reduce_chain passando o resultado anterior como param
pipeline = map_stage | prepare_reduce_input | reduce_chain

# prepare_map_inputs(parts) => return array com dicts de context contendo as partes do texto
# map_chain.map_prompt(array de dicts) + llm + StrOutputParser() => return array com os resumos (feitos pela llm) das partes
# prepare_reduce_input(summaries) => return dict com uma só chave context contendo os resumos numa string separada por \n
# reduce_chain.reduce_prompt(context) => recebe os resumos separados por \n e combina num único resumo
result = pipeline.invoke(parts)
print(result)
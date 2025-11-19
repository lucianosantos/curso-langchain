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

# MAP_REDUCE STRATEGY
# Phase 1: MAP - Summarize each chunk individually
print("=" * 80)
print("PHASE 1: MAP - Summarizing each chunk individually")
print("=" * 80)

map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Resuma o seguinte trecho de texto de forma concisa:\n\n{text}\n\nResumo:"
)

map_chain = map_prompt | llm | StrOutputParser()

summaries = []
for i, part in enumerate(parts, 1):
    print(f"\n--- Processing Chunk {i}/{len(parts)} ---")
    print(f"Chunk content (first 100 chars): {part.page_content[:100]}...")
    
    summary = map_chain.invoke({"text": part.page_content})
    summaries.append(summary)
    
    print(f"Summary: {summary}")

# Phase 2: REDUCE - Combine all summaries into a final summary
print("\n" + "=" * 80)
print("PHASE 2: REDUCE - Combining all summaries into final summary")
print("=" * 80)

reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="Combine os seguintes resumos em um único resumo coerente e completo em Português:\n\n{text}\n\nResumo Final:"
)

reduce_chain = reduce_prompt | llm | StrOutputParser()

# Join all individual summaries
combined_summaries = "\n\n".join([f"Resumo {i+1}: {summary}" for i, summary in enumerate(summaries)])

print(f"\nCombined summaries being sent to reduce phase:\n{combined_summaries}\n")

# Generate final summary
result = reduce_chain.invoke({"text": combined_summaries})

print("=" * 80)
print("FINAL RESULT:")
print("=" * 80)
print(result)
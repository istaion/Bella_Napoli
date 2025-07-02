import chromadb
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os 
os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"
# --- 1. Configuration ---

# Le nom de la collection que nous avons créée dans le script précédent.
COLLECTION_NAME = "la_belle_pizza_collection"
# Le modèle d'embedding (doit être le même que celui utilisé pour la création).
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
# Le modèle de LLM à utiliser pour la génération de la réponse.
LLM_MODEL = "mistral:7b"

print("Initialisation de ChromaDB...")
# Crée un client ChromaDB qui stockera les données sur le disque dans le dossier `chroma_db`
client = chromadb.PersistentClient(path="./data/chroma_db")

# --- 2. Initialisation des composants LangChain ---

print("Initialisation des composants LangChain...")

# Initialise le client Ollama pour les embeddings
ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialise le client ChromaDB pour se connecter à la base de données existante.
vectorstore = Chroma(
    client=chromadb.PersistentClient(path="./data/chroma_db"),
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings
)

# Crée un retriever à partir du vectorstore.
# Le retriever est responsable de la recherche des documents pertinents.
retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5
        }
    )

# Initialise le modèle de chat Ollama
llm = ChatOllama(model=LLM_MODEL,
        temperature=0.1)

# --- 3. Définition du prompt RAG ---

# Le template du prompt pour le LLM.
# Il inclut le contexte récupéré et la question de l'utilisateur.
template = """Tu es un expert du menu du restaurant italien VAPIANO.

## INFORMATIONS IMPORTANTES
- Réponds UNIQUEMENT avec les informations du contexte fourni
- Sois PRÉCIS sur les compositions et prix
- Pour les allergènes, traduis les numéros : 1=Gluten, 2=Crustacés, 3=Œufs, 4=Poissons, 5=Arachides, 6=Soja, 7=Lait, 8=Fruits à coque, 9=Céleri, 10=Moutarde, 11=Sésame, 12=Sulfites, 13=Mollusques, 14=Lupin
- Si une information manque, dis "Information non disponible dans le menu"

## CONTEXTE DU MENU
{context}

## QUESTION
{question}

## RÉPONSE STRUCTURÉE
Fournis une réponse claire avec :
- 🍽️ Nom du plat
- 🥄 Composition/Ingrédients  
- 💰 Prix
- ⚠️ Allergènes (si applicable)
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 4. Construction de la chaîne RAG avec LangChain Expression Language (LCEL) ---

# La chaîne RAG est construite en utilisant LCEL pour une meilleure lisibilité et modularité.
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. Boucle d'interaction ---

if __name__ == "__main__":
    print("\n--- Chatbot RAG avec LangChain ---")
    print("Posez des questions sur le document. Tapez 'exit' pour quitter.")

    while True:
        user_question = input("\nVous: ")
        if user_question.lower() == "exit":
            break

        print("Assistant: ...")
        # Invoque la chaîne RAG avec la question de l'utilisateur
        answer = rag_chain.invoke(user_question)
        print(f"\rAssistant: {answer}")

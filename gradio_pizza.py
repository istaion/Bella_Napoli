import gradio as gr
import re
import chromadb
import gradio as gr
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------- le modèle

COLLECTION_NAME = "la_belle_pizza_collection"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
LLM_MODEL = "llama3.1:8b"
CHROMA_PATH = "./data/chroma_db"

client = chromadb.PersistentClient(path=CHROMA_PATH)

ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=ollama_embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 8  # Booommmmm ! On charge !
    }
)

llm = ChatOllama(model=LLM_MODEL)

template = """Tu es un assistant spécialisé dans les informations sur le menu du restaurant VAPIANO.

## RÈGLES IMPORTANTES
- Réponds UNIQUEMENT avec les informations présentes dans le contexte fourni
- Pour les allergènes, sois TRÈS précis et mentionne TOUS les allergènes listés
- Si une information n'est pas dans le contexte, dis "Je n'ai pas cette information dans ma base de données"
- Utilise les noms EXACTS des plats tels qu'ils apparaissent dans le contexte

## CONTEXTE FOURNI
{context}

## QUESTION DU CLIENT
{question}

## RÉPONSE
Réponds de manière claire et précise en utilisant uniquement les informations du contexte ci-dessus."""

prompt = ChatPromptTemplate.from_template(template)

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# ------------- Bon la j'ai triché c'est l'ia :


def extract_dish_name(question):
    """
    Extrait le nom du plat de la question
    """
    # Cherche des patterns comme "pizza X" ou "je voudrais X"
    patterns = [
        r'pizza\s+([A-Z\s_]+?)(?:\s*[.?!]|$)',
        r'prendre\s+(?:la\s+)?(?:pizza\s+)?([A-Z\s_]+?)(?:\s*[.?!]|$)',
        r'voudrais\s+(?:la\s+)?(?:pizza\s+)?([A-Z\s_]+?)(?:\s*[.?!]|$)',
    ]
    
    question_upper = question.upper()
    
    for pattern in patterns:
        match = re.search(pattern, question_upper)
        if match:
            dish_name = match.group(1).strip()
            # Nettoie le nom
            dish_name = re.sub(r'\s+', ' ', dish_name)
            return dish_name
    
    return None

def hybrid_search(question, k=8):
    """
    Recherche hybride qui combine recherche vectorielle et recherche par mots-clés
    """
    print(f"\n🔍 RECHERCHE HYBRIDE: {question}")
    
    # 1. Recherche vectorielle standard
    docs = vectorstore.similarity_search(question, k=k)
    
    # 2. Extraction du nom du plat
    dish_name = extract_dish_name(question)
    print(f"🍕 Plat détecté: {dish_name}")
    
    # 3. Recherche spécifique par nom de plat si détecté
    if dish_name:
        # Essaie plusieurs variantes du nom
        dish_variants = [
            dish_name,
            dish_name.replace(' ', '_'),
            dish_name.replace('_', ' '),
            dish_name.replace('DI BUFALA', 'DI_BUFALA'),
            dish_name.replace('DI_BUFALA', 'DI BUFALA')
        ]
        
        for variant in dish_variants:
            print(f"🔎 Recherche variante: {variant}")
            variant_docs = vectorstore.similarity_search(
                f"allergènes {variant}",
                k=3,
                filter={"type": "allergenes"}
            )
            
            # Ajoute les nouveaux documents
            for doc in variant_docs:
                if doc not in docs and variant.upper() in doc.page_content.upper():
                    docs.insert(0, doc)  # Ajoute en priorité
                    print(f"✅ Document spécifique trouvé pour {variant}")
    
    # 4. Recherche spécifique allergènes si question d'allergie
    allergen_keywords = ['allergique', 'allergie', 'allergène', 'céleri', 'gluten', 'lait', 'œuf', 'soja']
    if any(keyword in question.lower() for keyword in allergen_keywords):
        print("🚨 Recherche allergènes activée")
        
        allergen_docs = vectorstore.similarity_search(
            f"allergènes pizza {question}",
            k=5,
            filter={"type": "allergenes"}
        )
        
        # Combine en évitant les doublons
        for doc in allergen_docs:
            if doc not in docs:
                docs.append(doc)
    
    # 5. Si c'est une question sur MARGHERITA DI BUFALA, force la récupération
    if 'margherita' in question.lower() and 'bufala' in question.lower():
        print("🎯 Recherche forcée MARGHERITA DI BUFALA")
        
        # Récupère TOUS les documents et filtre manuellement
        all_docs = vectorstore.similarity_search("MARGHERITA_DI_BUFALA allergènes céleri", k=15)
        
        for doc in all_docs:
            if "MARGHERITA_DI_BUFALA" in doc.page_content and "CÉLERI" in doc.page_content:
                # Ajoute en première position si pas déjà présent
                if doc not in docs:
                    docs.insert(0, doc)
                    print("✅ Document MARGHERITA_DI_BUFALA forcé en première position!")
                break
    
    # Limite à k documents maximum
    docs = docs[:k]
    
    print(f"📄 {len(docs)} documents finaux sélectionnés:")
    for i, doc in enumerate(docs):
        metadata = doc.metadata
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i+1}. {metadata.get('type', 'N/A')} ({metadata.get('source', 'N/A')}): {content_preview}...")
    
    return docs

def ask_rag(question):
    """
    Fonction RAG avec recherche hybride améliorée
    """
    try:
        if not question.strip():
            return "Veuillez poser une question."
        
        print(f"\n" + "="*60)
        print(f"QUESTION: {question}")
        
        # Recherche hybride
        docs = hybrid_search(question)
        
        # Formatage du contexte
        context_parts = []
        for doc in docs:
            metadata = doc.metadata
            source_info = f"[Source: {metadata.get('source', 'N/A')} - Type: {metadata.get('type', 'N/A')}]"
            context_parts.append(f"{source_info}\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        
        # Génération de la réponse
        formatted_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            final_response = response.content
        else:
            final_response = str(response)
        
        print(f"✅ RÉPONSE: {final_response[:200]}...")
        print("="*60)
        
        return final_response
    
    except Exception as e:
        error_msg = f"Erreur lors du traitement de votre question: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# ------------- l'interface de gradio

# def ask_rag(question):
#     try:
#         return rag_chain.invoke(question)
#     except Exception as e:
#         return f"Erreur : {str(e)}"

with gr.Blocks(title="Chatbot - La belle piza !") as demo:
    gr.Markdown("""
        **Vla la doc !**
        """)
    gr.File(value="./data/pdf/Menu.pdf", label="Menu", interactive=False)
    gr.File(value="./data/pdf/Liste_allergenes.pdf", label="Liste des allergènes", interactive=False)
    gr.Markdown("Posez une question sur le menu ou les allergènes.")
    gr.Markdown("ex : Bonjour, j'aime le fromage et les câpres, as tu un conseil ?")
    gr.Markdown("ex : Bonjour, je voudrais prendre la pizza MARGHERITA DI BUFALA. Je suis allergique au céleri c'est un soucis ? ")

    with gr.Row():
        chat_input = gr.Textbox(label="Votre question", placeholder="Quels plats ne contiennent pas de gluten ?")
        chat_output = gr.Textbox(label="Réponse du chatbot")

    submit_btn = gr.Button("Envoyer")

    submit_btn.click(fn=ask_rag, inputs=chat_input, outputs=chat_output)


demo.launch(server_name="0.0.0.0", server_port=7860)
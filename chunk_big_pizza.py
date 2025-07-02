import os
import chromadb
from chromadb.utils import embedding_functions
import fitz
import re
import json

# Désactive complètement la télémétrie pour éviter les erreurs
os.environ["CHROMA_ENABLE_TELEMETRY"] = "false"

# --- 1. Configuration ---
COLLECTION_NAME = "la_belle_pizza_collection"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"

def load_and_chunk_pdf(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Charge et découpe le PDF du menu en chunks
    """
    print(f"Chargement du fichier : {file_path}")
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    print(f"Le document contient {len(text)} caractères.")

    # Nettoyage basique du texte
    text = re.sub(r'\s+', ' ', text)  # Normalise les espaces
    text = text.strip()

    print("Découpage du texte en chunks...")
    chunks = []
    
    # Découpage par sections du menu
    sections = ["ANTIPASTI", "INSALATA", "PIZZA", "PASTA", "RISOTTO", "DOLCI", "EXTRAS", "BOISSONS", "KIDS"]
    
    for i, section in enumerate(sections):
        start_idx = text.upper().find(section)
        if start_idx != -1:
            # Trouve la fin de cette section (début de la suivante)
            end_idx = len(text)
            for next_section in sections[i+1:]:
                next_start = text.upper().find(next_section, start_idx + len(section))
                if next_start != -1:
                    end_idx = next_start
                    break
            
            section_text = text[start_idx:end_idx].strip()
            if section_text:
                chunks.append(f"Section {section}:\n{section_text}")
    
    # Si aucune structure détectée, découpage classique
    if not chunks:
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

    print(f"{len(chunks)} chunks ont été créés pour le menu.")
    return chunks

def load_allergenes_json(file_path):
    """
    Charge le fichier JSON des allergènes et le prépare pour l'indexation
    """
    print(f"Chargement du fichier JSON : {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    
    # Chunk pour les informations générales
    general_info = f"""Informations allergènes - Restaurant: {data['restaurant']}
Date de mise à jour: {data['date_mise_a_jour']}
Avertissement: {data['avertissement']}"""
    chunks.append(general_info)
    
    # Chunk pour chaque catégorie de produits
    for category, items in data['allergenes_par_produit'].items():
        if isinstance(items, dict):
            category_text = f"Allergènes - Catégorie {category}:\n"
            for item_name, allergenes in items.items():
                if isinstance(allergenes, list):
                    allergenes_str = ", ".join(allergenes)
                    category_text += f"- {item_name}: {allergenes_str}\n"
            chunks.append(category_text)
        elif isinstance(items, list):
            # Pour les items simples comme PAIN_CIABATTA
            allergenes_str = ", ".join(items)
            chunks.append(f"Allergènes - {category}: {allergenes_str}")
    
    # Chunk pour les recherches par allergène
    search_info = "Recherche par allergène:\n"
    for allergen_type, info in data['recherche_par_allergene'].items():
        search_info += f"\n{allergen_type}:\n"
        if 'note' in info:
            search_info += f"Note: {info['note']}\n"
        if 'plats_potentiels' in info and info['plats_potentiels']:
            search_info += f"Plats possibles: {', '.join(info['plats_potentiels'])}\n"
        if 'plats_possibles' in info:
            search_info += f"Plats possibles: {', '.join(info['plats_possibles'])}\n"
        if 'plats' in info:
            search_info += f"Plats: {', '.join(info['plats'])}\n"
    
    chunks.append(search_info)
    
    print(f"{len(chunks)} chunks ont été créés pour les allergènes.")
    return chunks

def main():
    print("🚀 Initialisation de ChromaDB...")
    # Crée le client sans télémétrie
    client = chromadb.PersistentClient(path="./data/chroma_db")

    print("🤖 Initialisation de la fonction d'embedding via Ollama...")
    try:
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=EMBEDDING_MODEL,
        )
    except Exception as e:
        print(f"❌ Erreur avec Ollama: {e}")
        print("Vérifiez que Ollama est lancé et que le modèle est téléchargé:")
        print(f"ollama pull {EMBEDDING_MODEL}")
        return

    print(f"📂 Création ou chargement de la collection : {COLLECTION_NAME}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=ollama_ef
    )

    # Traite le fichier PDF du menu
    menu_pdf_path = "data/pdf/Menu.pdf"
    if os.path.exists(menu_pdf_path):
        try:
            print("📋 Traitement du menu PDF...")
            menu_chunks = load_and_chunk_pdf(menu_pdf_path)
            
            if menu_chunks:
                print(f"💾 Stockage des chunks du menu...")
                ids = [f"chunk_menu_{i}" for i in range(len(menu_chunks))]
                metadatas = [{"source": "Menu.pdf", "type": "menu"} for _ in range(len(menu_chunks))]
                
                collection.add(
                    documents=menu_chunks, 
                    ids=ids,
                    metadatas=metadatas
                )
                print(f"✅ Menu traité avec succès!")
            else:
                print(f"⚠️  Aucun chunk créé pour le menu")
                
        except Exception as e:
            print(f"❌ Erreur lors du traitement du menu: {e}")
    else:
        print(f"⚠️  Fichier menu non trouvé: {menu_pdf_path}")

    # Traite le fichier JSON des allergènes
    allergenes_json_path = "data/json/allergene.json"
    if os.path.exists(allergenes_json_path):
        try:
            print("🚨 Traitement du fichier allergènes JSON...")
            allergenes_chunks = load_allergenes_json(allergenes_json_path)
            
            if allergenes_chunks:
                print(f"💾 Stockage des chunks des allergènes...")
                ids = [f"chunk_allergenes_{i}" for i in range(len(allergenes_chunks))]
                metadatas = [{"source": "allergene.json", "type": "allergenes"} for _ in range(len(allergenes_chunks))]
                
                collection.add(
                    documents=allergenes_chunks, 
                    ids=ids,
                    metadatas=metadatas
                )
                print(f"✅ Allergènes traités avec succès!")
            else:
                print(f"⚠️  Aucun chunk créé pour les allergènes")
                
        except Exception as e:
            print(f"❌ Erreur lors du traitement des allergènes: {e}")
    else:
        print(f"⚠️  Fichier allergènes non trouvé: {allergenes_json_path}")

    print(f"\n🎉 Base de données vectorielle créée avec succès !")
    print(f"📊 Nombre total de documents stockés : {collection.count()}")

if __name__ == "__main__":
    main()
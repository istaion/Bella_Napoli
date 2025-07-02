# Bella_Napoli

## Étape 1 : Ollama

Il faut pull mistral:7b et nomic-embed-text:v1.5

bash
```
ollama pull nomic-embed-text:v1.5
```
bash
```
ollama pull llama3.1:8b
```
(au moins l'embeding, pour le model il peux être changé dans gradio_pizza.py)

# Étape 2 : installation de uv (gestionnaire moderne)
bash
```
pip install --upgrade pip && \
pip install uv
```
# Étape 5 : installation des dépendances via uv
bash
```
uv pip install .
```

# Étape 6 : C'est partit pour le chat pizza !
bash
```python gradio_pizza.py
```

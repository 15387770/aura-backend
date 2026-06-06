import os
import anthropic
import chromadb
import hashlib
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
COLECCION  = "aura"
CARPETA_BD = "data/base_vectorial"

Path(CARPETA_BD).mkdir(parents=True, exist_ok=True)
cliente_ai = anthropic.Anthropic(api_key=API_KEY)
db         = chromadb.PersistentClient(path=CARPETA_BD)
coleccion  = db.get_or_create_collection(name=COLECCION)
historiales = {}

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def obtener_embedding(texto):
    h = hashlib.sha256(texto.encode()).hexdigest()
    return [int(h[i], 16) / 15.0 for i in range(64)]

def dividir_texto(texto, tamano=500, solapamiento=50):
    palabras = texto.split()
    frags, inicio = [], 0
    while inicio < len(palabras):
        frags.append(" ".join(palabras[inicio:inicio+tamano]))
        inicio += tamano - solapamiento
    return frags or [texto]

def agregar_informacion(texto, etiqueta="memoria"):
    frags = dividir_texto(texto)
    existente = coleccion.get()["ids"]
    ids, docs, metas, embs = [], [], [], []
    for i, frag in enumerate(frags):
        nid = f"{etiqueta}_{i}_{len(existente)+i}"
        if nid not in existente:
            ids.append(nid); docs.append(frag)
            metas.append({"fuente": etiqueta})
            embs.append(obtener_embedding(frag))
    if ids:
        coleccion.add(documents=docs, ids=ids, metadatas=met

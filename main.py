import os
import anthropic
import chromadb
import hashlib
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
COLECCION = "giia"
CARPETA_BD = "data/base_vectorial"

Path(CARPETA_BD).mkdir(parents=True, exist_ok=True)
cliente_ai = anthropic.Anthropic(api_key=API_KEY)
db = chromadb.PersistentClient(path=CARPETA_BD)
coleccion = db.get_or_create_collection(name=COLECCION)
historiales = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

def obtener_embedding(texto):
    h = hashlib.sha256(texto.encode()).hexdigest()
    return [int(h[i], 16) / 15.0 for i in range(64)]

def dividir_texto(texto, tamano=500, solapamiento=50):
    palabras = texto.split()
    frags = []
    inicio = 0
    while inicio < len(palabras):
        frags.append(" ".join(palabras[inicio:inicio+tamano]))
        inicio += tamano - solapamiento
    return frags or [texto]

def agregar_informacion(texto, etiqueta="memoria"):
    frags = dividir_texto(texto)
    existente = coleccion.get()["ids"]
    ids = []
    docs = []
    metas = []
    embs = []
    for i, frag in enumerate(frags):
        nid = f"{etiqueta}_{i}_{len(existente)+i}"
        if nid not in existente:
            ids.append(nid)
            docs.append(frag)
            metas.append({"fuente": etiqueta})
            embs.append(obtener_embedding(frag))
    if ids:
        coleccion.add(
            documents=docs,
            ids=ids,
            metadatas=metas,
            embeddings=embs
        )

def buscar_contexto(pregunta, n=4):
    total = coleccion.count()
    if total == 0:
        return ""
    res = coleccion.query(
        query_embeddings=[obtener_embedding(pregunta)],
        n_results=min(n, total)
    )
    return "\n\n".join(res["documents"][0])

class ChatRequest(BaseModel):
    session_id: str
    mensaje: str

class InfoRequest(BaseModel):
    texto: str
    etiqueta: str = "dato"

@app.get("/")
def root():
    return {"status": "GIIA activa"}

@app.post("/chat")
def chat(req: ChatRequest):
    historial = historiales.setdefault(req.session_id, [])
    contexto = buscar_contexto(req.mensaje)
    if contexto:
        sistema = (
            "Eres GIIA, una asistente personal inteligente, cercana y concisa. "
            "Tu nombre es GIIA y debes presentarte siempre como GIIA. "
            "Responde SOLO lo que te preguntan, en español, de forma natural y breve.\n\n"
            "Informacion del usuario:\n" + contexto
        )
    else:
        sistema = (
            "Eres GIIA, una asistente personal inteligente, cercana y concisa. "
            "Tu nombre es GIIA y debes presentarte siempre como GIIA. "
            "Responde SOLO lo que te preguntan, en español, de forma natural y breve. "
            "Aun no tienes informacion guardada del usuario."
        )
    historial.append({"role": "user", "content": req.mensaje})
    resp = cliente_ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=sistema,
        messages=historial
    )
    texto = resp.content[0].text
    historial.append({"role": "assistant", "content": texto})
    agregar_informacion(
        "Usuario: " + req.mensaje + "\nGIIA: " + texto,
        etiqueta="conversacion"
    )
    if len(historial) > 20:
        historial.pop(0)
        historial.pop(0)
    return {"respuesta": texto}

@app.post("/guardar")
def guardar(req: InfoRequest):
    agregar_informacion(req.texto, req.etiqueta)
    return {"status": "guardado"}

@app.get("/stats")
def stats():
    return {"fragmentos": coleccion.count()}

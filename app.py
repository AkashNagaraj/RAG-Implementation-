from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
import json

from RAG_system import main_rag 
from Smart_Agent import question_validator
from text_to_speech import main_TTS


app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ollama_model = Ollama(model="llama3.1")

@app.post("/api/TTS")
async def TTS(request: Request):
    try:
        body = await request.json()
        text = body["text"]
        response = main_TTS(text)
        return {"output":response}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/api/valid")
async def validator(request: Request):
    try:
        body = await request.json()
        question = body["question"]
        if "context" in body:
            context = body["context"]
            response =  question_validator(ollama_model, question, context)
        else:
            response = question_validator(ollama_model, question)
        return {"output":response}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/api/RAG")
async def hit_model(request: Request):
    try:
        body = await request.json()
        query = body["query"]
        print(query)
        response = main_rag(ollama_model, query)
        return {"output":response}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

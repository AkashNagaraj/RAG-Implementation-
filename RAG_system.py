import os
import chromadb
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


#model = Ollama(model = "llama3.1")


def chunk_data():
    pdf_path = "data/iesc111.pdf"
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_text = "\n".join([reader.pages[i].extract_text() for i in range(0,num_pages)])
    
    chunk_size = 1000
    overlap = 200 
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                    )
    chunk_text = text_splitter.split_text(pdf_text)

    ids = [str(i) for i in range(len(chunk_text))]
    print("Number of chunks in the pdf data:",len(chunk_text))
    return chunk_text, ids


def store_DB(text_content, ids):

    # Initialize Chroma DB
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="PDF", metadata={"hnsw:space": "cosine"})
    collection.add(documents=text_content, ids= ids)
    return collection


def query_chroma(collection, input_text):
    top_k = 3
    results = collection.query(query_texts=[input_text], n_results = top_k)
    required_output = " ".join(results["documents"][0])
    return required_output


def summarizer(model, text):
    result = model.invoke("Generate a concise summary for this : " + text)
    return result


def main_rag(model, input_text):
    text_data, ids = chunk_data()
    collection = store_DB(text_data, ids)
    text_data = query_chroma(collection, input_text)
    model_response = summarizer(model, text_data)
    print("Model response : ",model_response)
    return model_response


if __name__=="__main__":
    ollama_model = Ollama(model="llama3.1")
    main_rag(ollama_model, "What is sound")

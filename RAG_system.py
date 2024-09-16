import os
import chromadb
from langchain_community.llms import Ollama
from PyPDF2 import PdfReader


model = Ollama(model = "llama3.1")


def chunk_data():
    pdf_path = "data/iesc111.pdf"
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pdf_text = "\n".join([reader.pages[i].extract_text() for i in range(0,num_pages)])
    
    pdf_text = [val for val in pdf_text.split("\n") if val]
    chunk_size = 10 # Take 10 sentences for one chunk
    chunk_text = [" ".join(pdf_text[i:i+chunk_size]) for i in range(0,len(pdf_text)-chunk_size,chunk_size)]
    
    ids = [str(i) for i in range(len(chunk_text))]
    print("Number of chunks in the pdf data:",len(chunk_text))
    return chunk_text, ids


def store_DB(text_content, ids):

    # Initialize Chroma DB
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="PDF", metadata={"hnsw:space": "cosine"})

    # Add data to the collection
    collection.add(documents=text_content, ids= ids)
    return collection


def query_chroma(collection, input_text):
    top_k = 5
    results = collection.query(query_texts=[input_text], n_results = top_k)
    #print(results["documents"])
    required_output = " ".join(results["documents"][0])
    return required_output


def summary_text(input_text):
    summary = model.invoke("Summarize this text : " + input_text)
    return summary 


def main():
    input_text = input("Kwyword : ")
    text_data, ids = chunk_data()
    collection = store_DB(text_data, ids)
    text_data = query_chroma(collection, input_text)
    summary = summary_text(text_data)
    print(summary)


if __name__=="__main__":
    main()

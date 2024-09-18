from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import nltk
from nltk.tokenize import word_tokenize


#ollama_model = Ollama(model="llama3.1")


def model_initializations(context, question):
    # Initialize the Ollama model
    prompt_template = """
    You are a helpful assistant. Given the following context, answer the question with "Yes" or "No".

    Context: {context}

    Question: {question}

    Answer:
    """

    # Create a prompt template object
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt


def answer_question(ollama_model, context, question):
    prompt = model_initializations(context, question)
    formatted_prompt = prompt.format(context=context, question=question)
    qa_chain = LLMChain(llm=ollama_model, prompt=prompt)
    response = qa_chain.invoke({"context": context, "question": question})
    
    # Preprocess to extract the required answer
    output = [val.lower() for val in word_tokenize(response["text"])]
    if "yes" in output:
        return "Checking DB"
    else:
        return "Not valid question for the given context"


def question_validator(ollama_model, question, context = "Is the question a valid question that can be answered from an NCERT book?"):
    response = answer_question(ollama_model, context, question)
    return response

    
if __name__=="__main__":
    ollama_model = Ollama(model="llama3.1")
    question_validator(ollama_model, "How is sound propogated", context = "Is the question a valid question that can be answered from an NCERT book?")

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import nltk
from nltk.tokenize import word_tokenize


#ollama_model = Ollama(model="llama3.1")


def model_initializations(context, question):
    # Initialize the Ollama model
    prompt_template = """
    Is this question : {question} is related to this context :{context}. Give a Yes/ No answer. Nothing else.
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
    print("Model output:",output)

    if "yes" in output:
        return "yes"
    else:
        return "no"


def question_validator(ollama_model, question, context = "The PDF discusses the topic of sound, including its production, transmission, and characteristics. It explains that sound is produced by vibrating objects and travels as longitudinal waves through a medium (air, water, solids). Key concepts covered include how sound propagates, compressions and rarefactions, sound waves being mechanical waves, and their properties like frequency, amplitude, and speed. It also covers phenomena such as echoes, reverberation, the speed of sound in different media, the range of human hearing, and practical applications of sound such as ultrasound in medical and industrial fields. The text emphasizes the scientific principles behind sound, its measurement, and various real-life examples and activities for understanding sound waves."):
    response = answer_question(ollama_model, context, question)
    return response

    
if __name__=="__main__":
    ollama_model = Ollama(model="llama3.1")
    question_validator(ollama_model, "How is sound propogated", context = "Is the question a valid question that can be answered from an NCERT book?")

import torch, json, pickle
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

#torch.cuda.set_device(0)
#llm = Ollama(model="llama3")
memory = ConversationBufferMemory()


def run_chat(llm, user_query):

    template = """You are a chatbot. Answers the users questions clearly.
    
    Current conversation : 
    {history}
    Human : {input}
    AI :
    """
    
    prompt_template = PromptTemplate(input_variables=["history","input"],template=template)
    conversation = ConversationChain(
        llm=llm,
        prompt = prompt_template,
        verbose=False,
        memory=memory,
        )

    output = conversation.invoke(user_query)
    extracted_messages = conversation.memory.chat_memory.messages
    return output


if __name__=="__main__":
    run_chat()


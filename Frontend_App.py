import requests
import streamlit as st

st.title("APP")

tab1, tab2, tab3, tab4 = st.tabs(["Rag", "Validate", "Chat", "Audio Data"])

base_url = "http://localhost:8000"

with tab1:
    end_point = "/api/RAG"
    user_input1 = st.text_input("Enter your query (RAG):", key="rag_input")
    button1 = st.button("Submit (RAG)", key="rag_submit")

    if button1:
        st.write("User input entered")
        response = requests.post(base_url + end_point, json={"query": user_input1})
        if response.status_code == 200:
            output = response.json()
            st.write(output["output"])

with tab2:
    end_point = "/api/valid"
    user_input2 = st.text_input("Enter your query (Validate):", key="valid_input")
    button2 = st.button("Submit (Validate)", key="valid_submit")

    if button2:
        st.write("User input entered")
        response = requests.post(base_url + end_point, json={"query": user_input2})
        if response.status_code == 200:
            output = response.json()
            #st.write(output)
            st.write(output["output"])
        else:
            st.write("Error :",response.json())

with tab3:
    end_point = "/api/chat"
    user_input3 = st.text_input("Enter your query (Chat):", key="chat_input")
    button3 = st.button("Submit (Chat)", key="chat_submit")

    if button3:
        st.write("User input entered")
        response = requests.post(base_url + end_point, json={"query": user_input3})
        if response.status_code == 200:
            output = response.json()
            st.write(output["output"]["response"])

with tab4:
    end_point = "/api/TTS"
    user_input4 = st.text_input("Enter your query (Audio):",key="audio_input")
    button4 = st.button("Submit (Audio)", key="audio_submit")

    if button4:
        st.write("User input entered")
        response = requests.post(base_url + end_point, json={"text":user_input4})
        if response.status_code == 200:
            output = response.json()
            st.write(output["output"])

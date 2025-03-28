from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import time

# Page Configurations
st.set_page_config(page_title="JusticeBot", layout="wide")

# Custom Styles and Animations
st.markdown(
    """
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f9f9f9;
        }
        .header {
            text-align: center;
            padding: 10px;
            background-color: #004d40;
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header img {
            max-width: 100px;
            margin-bottom: 10px;
        }
        .header h1 {
            margin: 5px;
            font-size: 2.5rem;
        }
        .header p {
            margin: 5px;
            font-size: 1.2rem;
            font-style: italic;
        }
        .chat-container {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .user-message {
            text-align: right;
            background-color: #e0f7fa;
            border-radius: 12px;
            padding: 10px;
            margin: 10px 0;
            color: #00796b;
        }
        .assistant-message {
            text-align: left;
            background-color: #ffebee;
            border-radius: 12px;
            padding: 10px;
            margin: 10px 0;
            color: #c62828;
        }
        .reset-btn {
            background-color: #004d40;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
            border: none;
        }
        .reset-btn:hover {
            background-color: #00362e;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Legal Animation */
        .legal-animation {
            text-align: center;
            margin-top: 20px;
        }
        .scales {
            display: inline-block;
            position: relative;
            width: 100px;
            height: 100px;
            border-bottom: 4px solid #333;
        }
        .scales:before {
            content: '';
            position: absolute;
            width: 20px;
            height: 4px;
            background-color: #333;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        .scale-left, .scale-right {
            position: absolute;
            width: 40px;
            height: 10px;
            background-color: #333;
            border-radius: 50%;
            top: 20%;
            animation: swing 2s infinite alternate ease-in-out;
        }
        .scale-left {
            left: -60%;
        }
        .scale-right {
            right: -60%;
        }
        @keyframes swing {
            0% {transform: rotate(0deg);}
            100% {transform: rotate(20deg);}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Welcome Popup
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True

if st.session_state.first_visit:
    st.session_state.first_visit = False
    st.success("Welcome to LawGPT! Your trusted legal assistant for Indian Penal Code queries.")

# Header Section
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.markdown(
        """
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/1042/1042327.png" alt="Law Icon">
            <h1>JusticeBot</h1>
            <p>Your trusted legal assistant for Indian Penal Code queries.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Animation Section
st.markdown(
    """
    <div class="legal-animation">
        <div class="scales">
            <div class="scale-left"></div>
            <div class="scale-right"></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Reset Conversation Functionality
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.info("Conversation has been reset!")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load Model and Retriever
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1", model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"})
db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt Template
prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize TogetherAI LLM
TOGETHER_AI_API = os.environ["TOGETHER_AI"]
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.5,
    max_tokens=1024,
    together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display Chat Messages
for message in st.session_state.messages:
    with st.container():
        if message.get("role") == "user":
            st.markdown(f'<div class="user-message">{message.get("content")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message.get("content")}</div>', unsafe_allow_html=True)

# User Input
input_prompt = st.chat_input("Type your question here...")

if input_prompt:
    # Display User Message
    with st.container():
        st.markdown(f'<div class="user-message">{input_prompt}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": input_prompt})

    # Process Assistant Response
    with st.container():
        st.markdown('<div class="assistant-message">Thinking...</div>', unsafe_allow_html=True)
        with st.status("Processing..."):
            result = qa.invoke(input=input_prompt)

        full_response = ""
        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.markdown(f'<div class="assistant-message">{full_response}</div>', unsafe_allow_html=True)

# Reset Button
st.button("Reset Conversation", on_click=reset_conversation, use_container_width=True, type="primary")

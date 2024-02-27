from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain, StuffDocumentsChain
# from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

import google.auth
# from google.oauth2 import service_account
# import vertexai
# import google.generativeai as genai
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
# from langchain_google_vertexai import ChatVertexAI

from dotenv import load_dotenv
import streamlit as st
import os

# Load the environment variables
load_dotenv()

# google authorization
CREDENTIALS, PROJECT_ID = google.auth.default()


# LLM
@st.cache_resource
def model_init(model_type="gemini"):
    # To use Ollama you must first install the model of choice, then provide its name as a parameter
    if model_type == "ollama":
        model = Ollama(
            model="mistral:latest",  # Provide your ollama model name here
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler]),
            temperature=0.0,
        )

    # Initializing Gemini
    elif model_type == "gemini":
        model = VertexAI(
            model_name="gemini-pro",
            max_output_tokens="2500",
            temperature=0.05,
            top_p=0.8,
            top_k=40,
            verbose=True,
            streaming=True,
            project=PROJECT_ID,
            credentials=CREDENTIALS,
        )

    elif model_type == "bison":
        model = VertexAI(streaming=True, project=PROJECT_ID, credentials=CREDENTIALS)

    elif model_type == "gemini-ultra":
        model = VertexAI(
            model_name="gemini-ultra",
            max_output_tokens="2500",
            temperature=0.05,
            top_p=0.8,
            top_k=40,
            verbose=True,
            streaming=True,
            project=PROJECT_ID,
            credentials=CREDENTIALS,
        )

    print("Model Loaded")

    return model


# DONE create function for vector db creation
# Vector Database
@st.cache_resource
def vector_db_init():
    persist_directory = "./db/gemini/"  # Persist directory path
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = CohereEmbeddings(model = "multilingual-22-12") # multilingual embeddings from cohere
    embeddings = VertexAIEmbeddings(
        model_name="textembedding-gecko-multilingual",
        project=PROJECT_ID,
        credentials=CREDENTIALS,
    )

    # Document splitting, embedding and vector database loading
    # DOES NOT have to be done in every run, just once and after you can simply refer to the db
    if not os.path.exists(persist_directory):
        # Data Pre-processing
        pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)

        pdf_documents = pdf_loader.load()
        text_documents = text_loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)

        pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
        text_context = "\n\n".join(str(p.page_content) for p in text_documents)

        pdfs = splitter.split_text(pdf_context)
        texts = splitter.split_text(text_context)

        data = pdfs + texts

        print("Data Processing Complete")

        vectordb = Chroma.from_texts(
            data, embeddings, persist_directory=persist_directory
        )
        vectordb.persist()

        print("Vector DB Creating Complete\n")

    elif os.path.exists(persist_directory):
        vectordb = Chroma(
            persist_directory=persist_directory, embedding_function=embeddings
        )

        print("Vector DB Loaded\n")

    return vectordb.as_retriever()


@st.cache_resource
def chain_init(_model, _retriever):
    # DONE- conversational template
    template = """
    You are an expert on xyz.

    (explain further context here)

    (describe the input to the model)

    You will gather your knowledge and deliver a good response to the user

    (rules to follow in a list)

    (other things to remember)

    Below is the new message you received from the user:
    {question}

    Below is the information you have gathered:
    {context}

    Please provide a helpful answer to the user:
    """

    # template above
    question_prompt_template = PromptTemplate.from_template(template=template)

    # template for map_reduce to write summaries of each document individually
    combine_template = "Write a summary of the following text:\n\n{summaries}"
    combine_prompt_template = PromptTemplate.from_template(template=combine_template)

    # We create a qa chain with our llm, retriever, and memory
    # Use chain_type refine so we cna build off of different information,
    # in addition to being wary of our context window
    qa_chain = RetrievalQA.from_chain_type(
        llm=_model,
        chain_type="map_reduce",
        retriever=_retriever,
        verbose=True,
        chain_type_kwargs={
            "question_prompt": question_prompt_template,
            "combine_prompt": combine_prompt_template,
        },
    )

    return qa_chain


# Set the webpage title
st.set_page_config(
    page_title="Basic RAG QA",
    page_icon="🤖",
)

st.button(
    "?",
    on_click=None,
    help=(
        "This chatbot leverages Retrieval Augmented Generation. It uses a folder "
        " of PDFs and text files to create a knowledge base. When you "
        "ask a question, the AI searches this base to find relevant information. It "
        "combines your prompt with this information to generate an answer, providing "
        "context-driven responses. The AI is trained on a model that is multilingual, "
        "so it can understand and answer questions in both English and Spanish."
    ),
)

# Create a header element
st.header("Basic RAG QA")

language = st.selectbox("Response Language", ("English", "Spanish"))

with st.spinner("🚀 Vector DB loading..."):
    retriever = vector_db_init()

# selected_model = st.selectbox("Select model:",
#                               ("Gemini", "PaLM 2"))

with st.spinner("🚀 Model loading..."):
    model = model_init("gemini")

with st.spinner("🚀 Chain loading..."):
    chain = chain_init(model, retriever)


# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Welcome to the RAG Chatbot!""",
        }
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Question", key="user_input"):
    # Add our input to the session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)
    with st.spinner("⚡️ Thinking..."):
        # Pass our input to the llm chain and capture the final responses.
        # It is worth noting that the Stream Handler is already receiving the
        # streaming response as the llm is generating. We get our response
        # here once the llm has finished generating the complete response.
        if language == "English":
            response = chain.run(user_prompt)
        else:
            response = chain.run(user_prompt + "\n Respond in spanish.")

        # Add the response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)
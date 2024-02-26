from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_python_agent, create_csv_agent # tools that will be used for sql agent to reason for
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType # for agent type assignment
from langchain.agents.react.agent import create_react_agent
from langchain.agents import (AgentExecutor, Tool) # agent executor to execute agent Tool, to make tool out of agent, ConversationalAgent to create template
from langchain.memory import ConversationBufferMemory # chat memory for agentfrom langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.globals import set_verbose, set_debug

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_google_vertexai import VertexAI, ChatVertexAI, VertexAIModelGarden
import google.auth

from dotenv import load_dotenv
import json
import pandas as pd
import os

import streamlit as st


##### Setup #####

# Load the environment variables
load_dotenv()

# google authorization
CREDENTIALS, PROJECT_ID = google.auth.default()

# set langchain to verbose, debug true
set_verbose(True)
set_debug(False)

##### LLM #####
@st.cache_resource
def model_init(model_type = 'gemini', model_category = 'static'): 
    if model_category == 'static':
        # To use Ollama you must first install the model of choice, then provide its name as a parameter
        if(model_type == "mistral"):
            model = Ollama(
                        model="mistral:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )
        
        # Initializing Gemini
        elif model_type == "gemini":
            model = VertexAI(
                model_name = "gemini-pro",
                max_output_tokens = "2500",
                temperature = 0.0,
                verbose = True,
                streaming=True,
                project=PROJECT_ID,
                credentials=CREDENTIALS
            )

        elif model_type == "openai":
            model = OpenAI(temperature = 0.05, 
                        verbose = True,
                        api_key=os.getenv("OPENAI_API_KEY"),
                        streaming=True)
            
        elif model_type == "orca2":
            model = Ollama(
                        model="orca2:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )
        elif model_type == "llama":
            model = Ollama(
                        model="llama2:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )
            
    else:
        # To use Ollama you must first install the model of choice, then provide its name as a parameter
        if(model_type == "mistral"):
            model = ChatOllama(
                        model="mistral:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )
        
        # Initializing Gemini
        elif model_type == "gemini":
            model = ChatVertexAI(
                model_name = "gemini-pro",
                max_output_tokens = "2500",
                temperature = 0.0,
                verbose = True,
                streaming=True,
                project=PROJECT_ID,
                credentials=CREDENTIALS
            )

        elif model_type == "openai":
            model = ChatOpenAI(temperature = 0.05, 
                        verbose = True,
                        api_key=os.getenv("OPENAI_API_KEY"),
                        streaming=True)
            
        elif model_type == "orca2":
            model = ChatOllama(
                        model="orca2:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )
        elif model_type == "llama":
            model = ChatOllama(
                        model="llama2:latest",  # Provide your ollama model name here
                        temperature = 0.0,
                        )

    return model

##### Agent #####
def agent_init( _llm):
    ###### DEFINE DATA PATH HERE ######
    path = "/Users/marconardoneguerra/Desktop/Data Science/LLMs/LLMs Repo/PANDAS_Agent/data/futuristic_city_traffic.csv"

    csv_agent = create_csv_agent(llm=_llm,
                                 path= path,
                                 agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 )

    # Create the whole list of tools
    tools=[
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="Useful to manipulate dataframes. The dataframe you can call is named 'df'."
                        "You can also create plots with 'df'."
        ),
    ]

    response_schemas = [
        ResponseSchema(name="Final Answer", description="The natural language answer to the user's query"),
        ResponseSchema(
            name="plot",
            description="Write the python code for the plot if it is asked for. If not, leave blank.",
    ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    df = pd.read_csv(path)

    ##### Agent Creation #####
    prompt_template = PromptTemplate.from_template(
        """**INSTRUCTIONS**
        You have been given a dataset to analyze as requested by the user. 

        Please answer the user's request utilizing the tools below:
        {tools}
        
        Refer to the tools as follows: {tool_names}

        Use the following format:

        Question: the input question you must answer
        Thought: your thought process of what action should be taken
        Action: the tool that will be used to get the desired result
        Action Input: the input into the tool
        Observation: the result of the action and what it means
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        **CONTEXT**
        Here is the dataframe head for context:
        """+
        
        str(df.head())

        +"""

        **CHAT HISTORY**

        Chat History:
        {chat_history}

        **FINAL OUTPUT FORMATTING**
        {format_instructions}

        **QUESTION**

        Question: {input} 
        Thought: {agent_scratchpad}
        Action:""",
        partial_variables={"format_instructions":output_parser.get_format_instructions()},
    )

    # create zero shot agent (react agent)
    agent = create_react_agent(
        llm=_llm,
        tools=tools,
        prompt=prompt_template,
    )

    # Initiate memory which allows for storing and extracting messages
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key='output')

    ##### Agent Chain #####

    # Create an AgentExecutor which enables verbose mode and handling parsing errors
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, memory=memory)

    agent_chain.handle_parsing_errors = True

    return agent_chain, output_parser

##### Streamlit Code #####

# Set the webpage title
st.set_page_config(
    page_title="Dataset Q&A",
    page_icon="📊",
)

with st.sidebar:
    st.markdown("Please select your model and type")
    # get params for functions
    model_name = st.selectbox("Select model", ["openai"])
    model_category = st.selectbox("Select model type", ["static", "chat"])
    st.warning("Because of conflicts with streamlit, you must set your file path in the code.")

    # initialize model, db, agent on run
    if st.button(label="Run", help="This will initialize the model, database, and agent. It will also reset the chat interface."):
        with st.spinner("🚀 Loading model..."):
            st.session_state["llm"] = model_init(model_name, model_category)
            print(f"Switched to: {model_name}, {model_category}")

        with st.spinner("🦾 Loading agent..."):
            st.session_state["agent_chain"], st.session_state["output_parser"] = agent_init(st.session_state["llm"])

        if "messages" in st.session_state:
            st.session_state.messages = [
            {"role": "assistant", "content": "Welcome to the Data QA bot! Ask whatever questions you'd like!"}
        ]

# Create a header element
st.header("Dataset Q&A")
st.markdown("**PLEASE RUN SIDEBAR FIRST BEFORE RUNNING CHAT INTERFACE**")    

#### Create a chat interface ####

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.

##  Session States ##
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Data QA bot! Ask whatever questions you'd like!"}
    ]

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


## Chat Code ## 
# We take questions/instructions from the chat input to pass to the LLM
if user_input := st.chat_input("Question", key="input"):

    st.chat_message("user").markdown(user_input)

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
   
    # Add the response to the chat window
    with st.chat_message("assistant"):
        with st.spinner("⚡️ Thinking..."):
            # invoke chain with necessary input variables
            try:
                response = st.session_state["agent_chain"].invoke({"input" : user_input})
                response = st.session_state["output_parser"].parse(response)
            except:
                response = st.session_state["agent_chain"].invoke({"input" : user_input})

            # Add the response to messages session state
            try:
                st.session_state.messages.append(
                    {"role": "assistant", "content":json.loads(response["output"])[0]["Final Answer"]})
            except:
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["output"]})

            try:
                st.write(json.loads(response["output"])[0]["Final Answer"])
            except:
                st.write(response["output"])

            try:
                st.plotly_chart(json.loads(response["output"])[0]["plot"])
            except:
                print(None)
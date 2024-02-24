import sqlite3
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent, create_react_agent # creation of sql agent, initiates chain
from langchain_community.agent_toolkits import SQLDatabaseToolkit # tools that will be used for sql agent to reason for
from langchain.agents.agent_types import AgentType # for agent type assignment
from langchain.sql_database import SQLDatabase # SQLDatabase connection
from langchain.agents import (AgentExecutor, Tool) # agent executor to execute agent Tool, to make tool out of agent, ConversationalAgent to create template
from langchain_experimental.tools import PythonREPLTool # python interface to be used by agent for plotting
from langchain.memory import ConversationBufferMemory # chat memory for agentfrom langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate
from langchain.globals import set_verbose, set_debug

from langchain_community.chat_models import ChatOllama
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI, ChatVertexAI

import google.auth
from dotenv import load_dotenv
import os

import streamlit as st

##### Setup #####

# Load the environment variables
load_dotenv()

# google authorization
CREDENTIALS, PROJECT_ID = google.auth.default()

# set langchain to verbose, debug true
set_verbose(True)
set_debug(True)

##### LLM #####
@st.cache_resource
def model_init(model_type = 'gemini'):
    # To use Ollama you must first install the model of choice, then provide its name as a parameter
    if(model_type == "ollama"):
        model = ChatOllama(
                        model="mistral:latest",  # Provide your ollama model name here
                        temperature = 0.0
                    )
    
    # Initializing Gemini
    elif model_type == "gemini":
        model = ChatVertexAI(
            model_name = "gemini-pro",
            max_output_tokens = "2500",
            temperature = 0.05,
            top_p = 0.8,
            top_k = 40,
            verbose = True,
            streaming=True,
            project=PROJECT_ID,
            credentials=CREDENTIALS
        )

    return model

##### SQL Database #####
@st.cache_resource
def db_init(file, data_table_name):
    # Connect to the SQLite database
    connection = sqlite3.connect("data.db")

    # create dataframe from data
    df = pd.read_csv(file)

    # Convert DataFrame to a SQLite table
    df.to_sql(data_table_name, connection, if_exists='replace')

    # connect to local database
    db = SQLDatabase.from_uri('sqlite:///data.db')

    return db 

##### SQL Tool, PythonREPL Definitions #####
@st.cache_resource
def agent_init(_db, _llm, data_description):
    sql_agent = create_sql_agent(
        llm = _llm,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION, # does a reasoning step before acting but has no memory
        toolkit = SQLDatabaseToolkit(db=_db, llm=_llm), # toolkit to be used
    )

    # Define a description to suggest how to determine the choice of tool
    description = (
        "Useful when you require to answer analytical questions about super store sales. "
        "Use this more than the Python REPL tool if the question is not about plotting data."
        "If you do not know the name of the table you have to extract from, list table names in the db."
    )

    # Create a Tool object for the sql agent
    sql_tool = Tool(
        name="sql_tool",
        func=sql_agent.invoke,
        description=description,
    )

    python_description = (
        "Useful for plotting data with matplotlib. "
        "Use this more than the SQL tool if the question is about plotting data."
        "Output will be a matplotlib plot image to be embedded in final answer."
        "You can access the data at 'data.db' with the sqlite3 package"
    )

    # Create the whole list of tools
    tools = [PythonREPLTool(name = "python_repl", description=python_description), sql_tool]

    ##### Agent Creation #####
    prompt_template = ChatPromptTemplate.from_template(
        """We are statistically analyzing a dataset for a user by giving them natural language answers to their data analysis questions.

        Below is a description of the dataset:

        """ + data_description + """

        Please answer the user's request utilizing the tools below:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: your thought process of what action should be taken
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Do not transform the format above. It is your way of thinking through the problem.

        If there is no need to answer the question, or it is irrelevant to your job, please make your final answer something similar to: "Please ask a relevant query."
        
        Begin!

        Question: {input} 
        Thought: {agent_scratchpad}"""
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
        agent=agent, tools=tools, handle_parsing_errors=True, memory=memory
    )

    return agent_chain

##### Streamlit Code #####

# Set the webpage title
st.set_page_config(
    page_title="Dataset Q&A",
    page_icon="📊",
)

run = False

with st.sidebar:
    # get params for functions
    model_name = st.selectbox("Select model", ["gemini", "ollama"])
    data_file = st.file_uploader("Upload dataset", type=["csv"])
    data_description = st.text_area("Description")
    st.markdown("Dataset descirption should include a general description (e.g., the purpose of the data, what it is used for, etc.)")
    data_table_name = st.text_area("Table Name")
    st.markdown("The table name should be a descriptive name for the dataset. Could also just be the name of the file without the extension.")

    # initialize model, db, agent on run
    if st.button(label="Run", help="This will initialize the model, database, and agent. It will also reset the chat interface."):
        st.session_state["run"] = True

# Create a header element
st.header("Dataset Q&A")
st.markdown("**PLEASE RUN SIDEBAR FIRST BEFORE RUNNING CHAT INTERFACE**")

if "run" in st.session_state and st.session_state["run"] == True:
    with st.spinner("🚀 Loading model..."):
        st.session_state["llm"] = model_init(model_name)

    with st.spinner("💾 Loading database..."):
        st.session_state["db"] = db_init(data_file, data_table_name)

    with st.spinner("🦾 Loading agent..."):
        st.session_state["agent_chain"] = agent_init(st.session_state["db"], st.session_state["llm"], data_description)

    if "messages"  in st.session_state:
        st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Data QA bot! Ask whatever questions you'd like!"}
    ]
    
    if "agent_scratchpad" in st.session_state:
        st.session_state.agent_scratchpad = ""
    

#### Create a chat interface ####

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.

##  Session States ##
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Data QA bot! Ask whatever questions you'd like!"}
    ]
if "agent_scratchpad" not in st.session_state:
    st.session_state.agent_scratchpad = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


## Chat Code ## 
# We take questions/instructions from the chat input to pass to the LLM
if user_input := st.chat_input("Question", key="input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("⚡️ Thinking..."):
        # invoke chain with necessary input variables
        response = st.session_state["agent_chain"].invoke({"input": user_input, 
                                        "agent_scratchpad": "", 
                                        "chat_history": st.session_state.messages})

        print(response)

        if response["output"] is None:
            with st.chat_message("assistant"):
                st.markdown("Sorry, I don't know how to answer that")

        # add scratchpad for next iteration
        st.session_state.agent_scratchpad = response["agent_scratchpad"]

        # Add the response to messages session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response["output"]}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            # st.markdown(f"""
            # **SQL Output:**
            # ```
            # {response.tool_output['sql_tool']}
            # ```

            # **Visualization:**
            # {response.tool_output['python_repl']} 


            # {response.output})
            # """  # Assuming visualization output placed in 'python_repl'

            st.markdown(response["output"])
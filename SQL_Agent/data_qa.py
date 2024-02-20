import sqlite3
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
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



from dotenv import load_dotenv
import os

import streamlit as st

##### Setup #####

# load environment variables
load_dotenv()

# set langchain to verbose, debug true
set_verbose(True)
set_debug(True)

##### LLM #####

#initialize llm
# llm = ChatOpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY")) # switch to whatever llm you'd like
llm = ChatOllama(model = "mistral:latest", temperature=0.05) # switch to whatever llm you'd like

##### SQL Database #####

# Connect to the SQLite database
connection = sqlite3.connect("data.db")

# create dataframe from data
df = pd.read_csv("/Users/marconardoneguerra/Desktop/Data Science/LLMs/LLMs Repo/SQL_Agent/data/superstoredata.csv") # replace with your own directory

# Convert DataFrame to a SQLite table named "RetailSalesTable"
df.to_sql("SuperStoreData", connection, if_exists='replace')

# connect to local database
db = SQLDatabase.from_uri('sqlite:///data.db')

##### SQL Tool, PythonREPL Definitions #####

sql_agent = create_sql_agent(
    llm = llm,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION, # does a reasoning step before acting but has no memory
    toolkit = SQLDatabaseToolkit(db=db, llm=llm), # toolkit to be used
)

# Define a description to suggest how to determine the choice of tool
description = (
    "Useful when you require to answer analytical questions about super store sales. "
    "Use this more than the Python REPL tool if the question is about sales analytics,"
    "like 'How many invoices were there in 2010?' or 'sum the sales of the United Kingdom'. "
    "Try not to use clause in the SQL."
)

# Create a Tool object for the sql agent
sql_tool = Tool(
    name="sql_tool",
    func=sql_agent.invoke,
    description=description,
)

# Create the whole list of tools
tools = [PythonREPLTool(name = "python_repl"), sql_tool]

##### Agent Creation #####
prompt_template = ChatPromptTemplate.from_template(
    """We are statistically analyzing a dataset for a user by giving them natural language answers to their data analysis questions.
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

    
    These keywords must never be translated and transformed:
    - Action:
    - Thought:
    - Action Input:
    because they are part of the thinking process instead of the output.

    If there is no need to answer the question, or it is irrelevant to your job, please make your final answer something similar to: "Please ask a relevant query."
    
    Begin!

    Question: {input} 
    Thought: {agent_scratchpad}"""
)

# create zero shot agent (react agent)
agent = create_react_agent(
    llm=llm,
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

# Set the webpage title
st.set_page_config(
    page_title="Dataset Q&A",
    page_icon="📊",
)

##### Streamlit Code #####

# Create a header element
st.header("Dataset Q&A")
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
        response = agent_chain.invoke({"input": user_input, 
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
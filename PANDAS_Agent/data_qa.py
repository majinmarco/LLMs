from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent # tools that will be used for sql agent to reason for
from langchain.agents.agent_types import AgentType # for agent type assignment
from langchain.agents.react.agent import create_react_agent
from langchain.agents import (AgentExecutor, Tool) # agent executor to execute agent Tool, to make tool out of agent, ConversationalAgent to create template
# from langchain.memory import ConversationBufferMemory # chat memory for agentfrom langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.globals import set_verbose, set_debug

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_google_vertexai import VertexAI, ChatVertexAI
import google.auth

from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import os

import streamlit as st


##### Setup #####

# Load the environment variables
# Must define OPENAI_API_KEY here
load_dotenv()

# google authorization
# CREDENTIALS, PROJECT_ID = google.auth.default()

# set langchain to verbose, debug true
set_verbose(True)
set_debug(False)

##### LLM #####
@st.cache_resource
def model_init(): 
    # if model_category == 'static':
        # # To use Ollama you must first install the model of choice, then provide its name as a parameter
        # if(model_type == "mistral"):
        #     model = Ollama(
        #                 model="mistral:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )
        
        # # Initializing Gemini
        # elif model_type == "gemini":
        #     model = VertexAI(
        #         model_name = "gemini-pro",
        #         max_output_tokens = "2500",
        #         temperature = 0.0,
        #         verbose = True,
        #         streaming=True,
        #         project=PROJECT_ID,
        #         credentials=CREDENTIALS
        #     )

        # elif model_type == "openai":
    model = OpenAI(temperature = 0.05,
                model = "gpt-3.5-turbo-instruct", 
                verbose = True,
                api_key=os.getenv("OPENAI_API_KEY"),
                streaming=False)
            
        # elif model_type == "orca2":
        #     model = Ollama(
        #                 model="orca2:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )
        # elif model_type == "llama":
        #     model = Ollama(
        #                 model="llama2:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )
            
    # else:
        # To use Ollama you must first install the model of choice, then provide its name as a parameter
        # if(model_type == "mistral"):
        #     model = ChatOllama(
        #                 model="mistral:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )
        
        # # Initializing Gemini
        # elif model_type == "gemini":
        #     model = ChatVertexAI(
        #         model_name = "gemini-pro",
        #         max_output_tokens = "2500",
        #         temperature = 0.0,
        #         verbose = True,
        #         streaming=True,
        #         project=PROJECT_ID,
        #         credentials=CREDENTIALS
        #     )

        # elif model_type == "openai":
        # model = ChatOpenAI(temperature = 0.05,
        #             model = "gpt-3.5-turbo-instruct",
        #             verbose = True,
        #             api_key=os.getenv("OPENAI_API_KEY"),
        #             streaming=True)
            
        # elif model_type == "orca2":
        #     model = ChatOllama(
        #                 model="orca2:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )
        # elif model_type == "llama":
        #     model = ChatOllama(
        #                 model="llama2:latest",  # Provide your ollama model name here
        #                 temperature = 0.0,
        #                 )

    return model

##### Agent #####
def agent_init( _llm, path):
    # init csv agent
    csv_agent = create_csv_agent(llm=_llm,
                                 path= path,
                                 agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                 )

    # Create the whole list of tools
    tools=[
        Tool(
            name="CSVAgent",
            func=csv_agent.invoke,
            description="Useful to write python code with which you can manipulate a dataframe named 'df' and plot graphs."
        ),
    ]

    response_schemas = [
        ResponseSchema(name="output", description="The natural language answer to the user's query"),
        ResponseSchema(
            name="plot",
            description="Write the python code for a graph that is relevant to the question using matplotlib.pyplot or pandas. If you cannot think of anything, leave this blank.",
    ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # output_fixer = OutputFixingParser.from_llm(parser=output_parser, llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

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
        Action: stating the tool that will be used to get the desired result
        Action Input: the input into the tool
        Observation: the result of the action and what it means
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: 
        {{
            "output":"the Final Answer to the user's question",
            "plot":"the python code for a plot that is relevant to the question"
        }}

        **CONTEXT**
        Here is the dataframe head for context:
        """+
        
        str(df.head())

        # **CHAT HISTORY**

        # Chat History:
        # {chat_history}

        +"""

        **QUESTION**

        Question: {input} 
        Thought: {agent_scratchpad}
        Action:""",
        #partial_variables={"format_instructions":output_parser.get_format_instructions()},
    )

    print(output_parser.get_format_instructions())

    # create zero shot agent (react agent)
    agent = create_react_agent(
        llm=_llm,
        tools=tools,
        prompt=prompt_template,
    )

    # Initiate memory which allows for storing and extracting messages
    #memory = ConversationBufferMemory(memory_key="chat_history", input_key="input", output_key='output')

    ##### Agent Chain #####

    # Create an AgentExecutor which enables verbose mode and handling parsing errors
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors = True)

    # chains pipeline together
    # full_chain = agent_chain #| output_parser

    return agent_chain, output_parser, df

# matplotlib figure display
@st.cache_resource
def execute_and_display_figure(code_string, path):
    # Path where you want to save the figure
    figure_path = "generated_figure.png"

    # Create a Python script from the provided code
    with open("temp_script.py", "w") as f:
        f.write(f"import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                f"df = pd.read_csv('{path}')\n")
        f.write(code_string)  # Write the code to the file
        f.write(f"plt.savefig('{figure_path}')")

    # Execute the script
    os.system("python temp_script.py")

    # Load and display the figure
    # (Replace this with how you display images in your application)
    image = plt.imread(figure_path)
    return image

##### Streamlit Code #####

# Set the webpage title
st.set_page_config(
    page_title="Dataset Q&A",
    page_icon="📊",
)

# with st.sidebar:
    # st.markdown("Please select your model and type")
    # # get params for functions
    # model_name = st.selectbox("Select model", ["openai"])
    # model_category = st.selectbox("Select model type", ["static", "chat"])
    # st.warning("Because of conflicts with streamlit, you must set your file path in the code.")

    # # initialize model, db, agent on run
    # if st.button(label="Run", help="This will initialize the model, database, and agent. It will also reset the chat interface."):
    #     with st.spinner("🚀 Loading model..."):
    #         st.session_state["llm"] = model_init(model_name, model_category)
    #         print(f"Switched to: {model_name}, {model_category}")

    #     with st.spinner("🦾 Loading agent..."):
    #         st.session_state["agent_chain"] = agent_init(st.session_state["llm"])

    #     if "messages" in st.session_state:
    #         st.session_state.messages = [
    #         {"role": "assistant", "content": "Welcome to the Data QA bot! Ask whatever questions you'd like!"}
    #     ]

# Create a header element
st.header("Dataset Q&A")   

###### DEFINE DATA PATH HERE ######
st.session_state["path"] = "/Users/marconardoneguerra/Desktop/Data Science/LLMs/LLMs Repo/PANDAS_Agent/data/futuristic_city_traffic.csv"

# Load everything in
with st.spinner("🚀 Loading model..."):
    st.session_state["llm"] = model_init()
    #print(f"Switched to: {model_name}, {model_category}")

with st.spinner("🦾 Loading agent..."):
    st.session_state["agent_chain"], st.session_state["output_parser"], st.session_state["df"] = agent_init(st.session_state["llm"], st.session_state["path"])

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
            response = st.session_state["agent_chain"].invoke({"input" : user_input})

            # parse output: "output" key
            response = st.session_state["output_parser"].parse(response['output'])

            # Add the response to messages session state
            st.session_state.messages.append(
                {"role": "assistant", "content":response["output"]})

            # Render the response
            st.write(response["output"])

            # Render the plot
            df = st.session_state["df"]

            try:
                # get plot image and display it
                st.image(execute_and_display_figure(response["plot"], st.session_state["path"]))
            except:
                st.write("No plot generated.")
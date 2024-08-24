from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# functions for the tools
def greet_user(name: str) -> str:
    """greets the user by name
    """
    return f'Hello , {name}' 

def reverse_string(text: str) -> str:
    """reverses the given string
    """
    return text[::-1]

def concatenate_strings(a: str, b: str) -> str:
    """concatenates the given string
    """
    return a + b

# pydantic model for tool arguments
class ConcatenateStringArgs(BaseModel):
    a: str = Field(description='First string')
    b: str = Field(description='Second string')

class GreetUserArgs(BaseModel):
    name: str = Field(description='Name of the user to greet')

class ReverseStringArgs(BaseModel):
    text: str = Field(description='string you want to reverse')

# create tools using Tool and StructuredTool constructor approach
tools = [
    StructuredTool(
        name='GreetUser',
        func=greet_user,
        description='Greets the user by name',
        args_schema=GreetUserArgs
    ),
    StructuredTool(
        name='ReverseString',
        func=reverse_string,
        description='Reverses the given string',
        args_schema=ReverseStringArgs
    ),
    # use StructuredTool for more complex functions that require multiple input parameters
    # StructuredTool allows us to define an input schema using Pydantic, ensuring proper validation and description
    StructuredTool.from_function(
        func=concatenate_strings,
        name='ConcatenateStrings',
        description='Concatenates two strings',
        args_schema=ConcatenateStringArgs # schema defining tool's input argument
    )
]
  
llm = ChatOpenAI(model='gpt-4o-mini')

prompt = hub.pull('hwchase17/openai-tools-agent')

# create ReAct agent using the create_tool_calling_agent function
agent = create_tool_calling_agent(
    llm=llm, # the agent to execute
    tools=tools, # list of tools available to the agent
    prompt=prompt # prompt template to guide the agent's responses
)

# create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, # the agent to execute
    tools=tools, # list of tools available to the agent
    verbose=True, # enable verbose logging
    handle_parsing_errors=True # handle parsing errors gracefully
)

# test the agent with sample queries
response = agent_executor.invoke({'input': 'Greet Alice'})
print('Response for "Greet Alice":', response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({'input': 'concatenate "hello" and "world"'})
print('Response for "concatenate hello and world":', response)
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# pydantic model for tool arguments
class ConcatenateStringArgs(BaseModel):
    a: str = Field(description='First string')
    b: str = Field(description='Second string')

class GreetUserArgs(BaseModel):
    name: str = Field(description='Name of the user to greet')

class ReverseStringArgs(BaseModel):
    text: str = Field(description='string you want to reverse')

# functions for the tools
@tool(args_schema=GreetUserArgs)
def greet_user(name: str) -> str:
    """greets the user by name
    """
    return f'Hello , {name}' 

@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """reverses the given string
    """
    return text[::-1]

@tool(args_schema=ConcatenateStringArgs)
def concatenate_strings(a: str, b: str) -> str:
    """concatenates the given string
    """
    return a + b

tools = [
    greet_user,
    reverse_string,
    concatenate_strings
]

llm = ChatOpenAI(model='gpt-4')

# Pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the ReAct agent using the create_tool_calling_agent function
# This function sets up an agent capable of calling tools based on the provided prompt.
agent = create_tool_calling_agent(
    llm=llm,  # Language model to use
    tools=tools,  # List of tools available to the agent
    prompt=prompt,  # Prompt template to guide the agent's responses
)

# create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)
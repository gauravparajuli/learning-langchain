import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description='First number to multiply')
    y: float = Field(description='Second number to multiply')

class MultiplyNumbersTool(BaseTool):
    name = 'multiply_numbers'
    description = 'useful for multiplying two numbers'
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(
        self,
        x: float,
        y: float
    ) -> str:
        """Use the tool """
        result = x * y
        return f'the product of {x} and {y} is {result}'
    
tools = [MultiplyNumbersTool()]

llm = ChatOpenAI(model='gpt-4o-mini')

prompt = hub.pull('hwchase17/openai-tools-agent')

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True, # see how the agent is thinking
    handle_parsing_errors=True
)

# test the agent with sample queries
response = agent_executor.invoke(dict(input='Search for Apple Intelligence'))
print('Response for "Search for Apple Intelligence"', response)

response = agent_executor.invoke({'input': 'Multiply 10 and 20'})
print('Response for "Multiply 10 and 20:"', response)
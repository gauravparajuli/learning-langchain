from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

# define a very simple tool that returns current time
def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()

    return now.strftime('%I:%M %p') # H:MM AM/PM format

# list of tools available to agent
tools = [
    Tool(
        name='Time',
        func=get_current_time,
        description='useful when you need to know the current time'
    )
]

# pull the prompt template from the hub
# ReAct = Reason and action
prompt = hub.pull('hwchase17/react')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# create ReAct agent using the create_react_agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True
)

# create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# run the agent with test query
response = agent_executor.invoke({'input': 'what time is it?'})

# print the response from the agent
print(f'{response=}')

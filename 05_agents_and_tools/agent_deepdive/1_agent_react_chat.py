from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

# define tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """searches wikipedia and returns the summary of first result
    """    
    from wikipedia import summary

    try:
        # limit to 2 sentences for brevity
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that"
    
# define tools that agent can use
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name='Wikipedia',
        func=search_wikipedia,
        description='Useful for when you need to know information about a topic'
    )
]

# load Chat Prompt from the hub
prompt = hub.pull('hwchase17/structured-chat-agent')

llm = ChatOpenAI(model='gpt-4o-mini')

# create a structured chat agent with conversation buffer memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True
)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# it combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# agent executor is responsible for managing the interaction between user input, the agent, and the tools
# it also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory, # use the conversation memory to maintain the context
    handle_parsing_errors=True # handle any parsing errors gracefully
)

# initial sysetm message to set the context for the chat
initial_message = """You are an AI assistant that can provide helpful answers using availabe tools
If you are unable to answer, you can use the following tools: Time and Wikipedia
"""
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# chat loop to interact with the user
while True:
    user_input = input('User: ')
    if user_input.lower() == 'exit':
        break

    # add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # invoke the agent with the user input and the current chat history
    response = agent_executor.invoke(dict(input=user_input))
    print(f"Bot: {response['output']}")

    # add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response['output']))
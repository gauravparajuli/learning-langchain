from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# load api key
load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

# system message -> message for priming AI behavior
# human message -> message from human to AI model
# ai message -> message from AI

messages = [
    SystemMessage(content='solve the following math problems'),
    HumanMessage(content='what is 81 divided by 27')
]

result = model.invoke(messages)
print(f'answer from AI: {result.content}')

messages = [
    SystemMessage(content='solve the following math problems'),
    HumanMessage(content='what is 81 divided by 9?'),
    AIMessage(content='81 divided by 9 is 9.'),
    HumanMessage(content='what is 10 times 5?')
]

result = model.invoke(messages)
print(f'answer from AI: {result.content}')
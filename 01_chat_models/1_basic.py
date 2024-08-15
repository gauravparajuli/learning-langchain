from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# load api key
load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

result = model.invoke('What is the average age of universe?')
print(f'Full result:\n', result)
print(f'Content only:\n', result.content)
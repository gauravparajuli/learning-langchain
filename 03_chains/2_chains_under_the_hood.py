from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a comedian who tells jokes about {topic}'),
        ('human', 'tell me {joke_count} jokes.')
    ]
)

# create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create the runnable sequence (equivalent to LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# run the code
response = chain.invoke(dict(topic='lawyers', joke_count=3))

print(response)
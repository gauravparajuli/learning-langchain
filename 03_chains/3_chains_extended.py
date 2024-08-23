from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a comedian who tells jokes about {topic}'),
        ('human', 'tell me {joke_count} jokes.')
    ]
)

# additional processing steps using runnablelambda
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f'word count: {len(x.split())}\n{x}')

# create combined chain using LCEL
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke(dict(topic='lawyers', joke_count=3))

print(result)
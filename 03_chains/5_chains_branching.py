from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

# define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant'),
        ('human', 'generate a thank you note for this positive feedback: {feedback}')
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant'),
        ('human', 'generate a response addressing this negative feedback: {feedback}')
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant'),
        ('human', 'generate a request for more details for this neutral feedback: {feedback}')
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant'),
        ('human', 'generate a message to escalate this feedback to a human agent: {feedback}')
    ]
)

# define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant.'),
        ('human', 'classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}')
    ]
)

# define runnable branches for handling the feedback
branches = RunnableBranch(
    (
        lambda x: 'positive' in x, positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'negative' in x, negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: 'neutral' in x, neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

# create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# combine classification and response generation into one chain
chain = classification_chain | branches

review = 'The product is terrible. it just broke after a single use'
result = chain.invoke(dict(feedback= review))

print(result)
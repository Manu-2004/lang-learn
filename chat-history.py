from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

load_dotenv()

llm = ChatOpenAI(temperature=0.8)

history = ChatMessageHistory()
history.add_user_message("Hello, Let's talk about GPTs")
history.add_ai_message("Sure, what would you like to know?")
dicts = messages_to_dict(history.messages)
new_messages = messages_from_dict(dicts)

history = ChatMessageHistory(messages=new_messages)
buffer = ConversationBufferMemory(chat_memory=history)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=buffer
)

print(conversation.predict(input = "What are they?"))
print(conversation.memory)


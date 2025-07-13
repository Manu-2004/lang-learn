from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0.8)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)



print("\nWelcome to your AI chatbot! Type 'bye' to exit.")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.strip().lower() == "bye":
        print("AI: Goodbye! 👋")
        print("=" * 50)
        break

    result = conversation.invoke(user_input)
    print(f"AI: {result['response']}")
    print("=" * 50)
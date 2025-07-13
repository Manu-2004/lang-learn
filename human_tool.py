from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.human.tool import HumanInputRun
from langgraph.prebuilt import create_react_agent   

load_dotenv()


llm = ChatOpenAI(temperature=0.8)


tools = [HumanInputRun()]

#  Agent (LangGraph pre‑built ReAct)
agent_executor = create_react_agent(llm, tools)


question = input("Enter your question: ")

# invoke() is the new, non‑deprecated entry‑point
result = agent_executor.invoke(
    {"messages": [{"role": "user", "content": question}]}
)

# The result is a dict whose last message is the agent’s reply
print(result["messages"][-1].content)

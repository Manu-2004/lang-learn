from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain.chains.llm_math.base import LLMMathChain


from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_chat_planner,
    load_agent_executor,
)


# Environment & LLM
load_dotenv()                              
llm = ChatOpenAI(temperature=0.8)


# Tools
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
wikipedia      = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
search         = SerpAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for general knowledge queries.",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful for current‑events queries.",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for math.",
    ),
]


# Planner & Executor
planner  = load_chat_planner(llm)                      # LLM that writes the plan
executor = load_agent_executor(llm, tools, verbose=True)  # executes each step

agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
)


question = (
    "Where are the next summer Olympics being held? "
    "What is the population of that country raised to the power 0.43?"
)

# PlanAndExecute implements the Runnable interface:
answer = agent.invoke(question)      # or agent.run(question) for older versions
print(answer["output"])              # final consolidated response

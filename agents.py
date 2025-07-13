from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun



from langchain.agents import initialize_agent, AgentType

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

prompt = input("Enter your question: ")

# Initialize LLM
llm = ChatOpenAI(temperature=0.8)



wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


tools1 = [wikipedia_tool]
tools2 =[DuckDuckGoSearchRun()]


agent = initialize_agent(
    tools1,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run(prompt)
print(result)




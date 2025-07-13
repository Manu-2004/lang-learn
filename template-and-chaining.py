from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(temperature=0.8)

# Prompt template
template = (
    "Consider you are a venture capitalist. You are asked to fund this product: {product}. "
    "Will you accept or reject considering the current situations in the market? Give your answer as 2-3 sentences"
)
prompt = PromptTemplate.from_template(template)

# Modern LangChain chain (Expression Language style)
chain = prompt | llm
# 🔄 Get user input
user_product = input("Enter a product idea to evaluate: ")

# Run the chain
response = chain.invoke({"product": user_product})
print("=" * 80)
print("📢 AI VC Decision")
print("=" * 80)
print(response.content)
print("=" * 80)


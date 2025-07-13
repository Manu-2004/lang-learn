from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(temperature=0.8)

template1 = "Suggest a creative and catchy company name for the following product: {product}"

prompt1 = PromptTemplate.from_template(template1)

first_chain = prompt1 | llm

template2 = "Come up with a catchy phrase or slogan for the company named: {company_name}"


prompt2 = PromptTemplate.from_template(template2)

second_chain = prompt2 | llm

product_idea = input("Enter a product idea: ")

company_name = first_chain.invoke({"product": product_idea}).content

slogan = second_chain.invoke({"company_name": company_name}).content

print(f"\n🏢 Company Name: {company_name}")
print(f"💬 Slogan: {slogan}")



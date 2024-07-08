from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4")
prompt = PromptTemplate(input_variables=["country"], template="{country}의 수도는 어디야?")

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke("대한민국"))

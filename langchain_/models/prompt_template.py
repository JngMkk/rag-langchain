from langchain_core.prompts import PromptTemplate

template = "{product}를 홍보하기 좋은 문구를 추천해줘?"

prompt = PromptTemplate(input_variables=["product"], template=template)
print(prompt.format(product="카메라"))

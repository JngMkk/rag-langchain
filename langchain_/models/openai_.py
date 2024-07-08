from langchain_openai import ChatOpenAI

# * temperature: 창의성
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

prompt = "나는 강아지를 키우고 있습니다. 내가 키우고 있는 동물은?"
print(llm.invoke(prompt))

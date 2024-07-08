from langchain.chains.conversation.base import ConversationChain
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4")
conversation = ConversationChain(llm=llm, verbose=True)

conversation.predict(input="진희는 강아지를 한 마리 키우고 있습니다.")
conversation.predict(input="영수는 고양이를 두 마리 키우고 있습니다.")
print(conversation.predict(input="진희와 영수가 키우는 동물은 총 몇 마리 일까요?"))
"""
진희 씨가 한 마리의 강아지를 키우고 있고, 영수 씨가 두 마리의 고양이를 키우고 있으므로, 진희와 영수가 키우는 동물은 총 세 마리입니다.
"""

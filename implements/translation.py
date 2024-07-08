import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

langs = ["Korean", "Japanese", "Chinese", "English"]
left_co, cent_co, last_co = st.columns(3)

with st.sidebar:
    language = st.radio("번역을 원하는 언어를 선택해주세요. :", langs)

st.markdown("### 언어 번역 서비스")
prompt = st.text_input("번역을 원하는 텍스트를 입력하세요.")

trans_template = PromptTemplate(
    input_variables=["trans"],
    template="Your task is to translate this text to " + language + "\n\nTEXT: {trans}",
)

# * 텍스트 저장 용도
memory = ConversationBufferMemory(input_key="trans", memory_key="chat_history")
llm = ChatOpenAI(temperature=0, model="gpt-4")
trans_chain = LLMChain(
    llm=llm, memory=memory, prompt=trans_template, output_key="translate", verbose=True
)

if st.button("번역"):
    if prompt:
        response = trans_chain({"trans": prompt})
        st.info(response["translate"])

import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.set_page_config(page_title="질문하세유~😀")
st.title("질문하세유~😀")


def generate_response(query: str) -> str:
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    return st.info(llm.invoke(query).content)


with st.form("query_form"):
    text = st.text_area("질문 입력: ", "What types of text models does OpenAI provide?")
    submmited = st.form_submit_button("질문하기")
    generate_response(text)

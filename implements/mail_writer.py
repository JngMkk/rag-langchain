import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

st.set_page_config(page_title="이메일 작성 서비스", page_icon=":robot:")
st.header("이메일 작성기")


def get_email():
    input_text = st.text_area(
        label="메일 입력",
        label_visibility="collapsed",
        placeholder="당신의 메일은...",
        key="input_text",
    )
    return input_text


def load_language_model():
    return ChatOpenAI(temperature=0, model="gpt-4")


def init_prompt_template(input_variables, query_template):
    return PromptTemplate(input_variables=input_variables, template=query_template)


QUERY_TEMPLATE = """
아래 메일 회신 예시를 작성해 주세요.
아래는 회신해야 할 이메일 입니다.
이메일: {email}

당신은 아래 조건을 만족해야 합니다.
    - 모든 언어는 한국어로 작성할 것.
    - 이메일은 공식적인 양식을 따를 것.
    - 이메일은 존중과 예의를 지켜 작성할 것.
    - 이메일은 누구나 이해할 수 있도록 작성할 것.
"""

input_text = get_email()
prompt = init_prompt_template("email", QUERY_TEMPLATE)
llm = load_language_model()

st.button("예제를 보여주세요.", type="secondary", help="봇이 작성한 메일을 확인해보세요.")
st.markdown("### 봇이 작성한 메일은:")

if input_text:
    prompt_with_email = prompt.format(email=input_text)
    formatted_email = llm.invoke(prompt_with_email)
    st.write(formatted_email.content)

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from PyPDF2 import PdfReader


def get_pdf_text(pdf_docs):
    """PDF 문서에서 텍스트 추출"""

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    """지정된 조건에 따라 주어진 텍스트를 더 작은 덩어리로 분할"""

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(chunks):
    """주어진 텍스트 청크에 대한 임베딩을 생성하고 FAISS를 사용하여 벡터 저장소 생성"""

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents


def get_conversation_chain(vector_store: FAISS):
    """주어진 벡터 저장소로 대화 체인을 초기화"""

    # * ConversationBufferWindMemory에 이전 대화 저장
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

    # * ConversationalRetrievalChain을 통해 랭체인 챗봇에 쿼리 전송
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        retriever=vector_store.as_retriever(),
        memory=memory,
        get_chat_history=lambda h: h,
    )

    return conversation_chain


def main():
    user_uploads = st.file_uploader("파일을 업로드해주세요", accept_multiple_files=True)
    if user_uploads is not None:
        if st.button("Upload"):
            with st.spinner("처리중..."):
                raw_text = get_pdf_text(user_uploads)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversation_chain(vector_store)

    user_query = st.chat_input("질문을 입력하세요.")
    if user_query:
        if "conversation" in st.session_state:
            response = st.session_state.conversation.run(
                question=user_query, chat_history=st.session_state.get("chat_history", [])
            )
        else:
            response = "문서 먼저."

        with st.chat_message("assistant"):
            st.write(response)


if __name__ == "__main__":
    main()

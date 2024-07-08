import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from PyPDF2 import PdfReader


def process_text(text):
    # * CharacterTextSplitter를 사용하여 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def process_embeds(chunks):
    # * HuggingFaceEmbeddings를 사용하여 청크를 임베딩
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    documents = FAISS.from_texts(chunks, embeddings)
    return documents


def main():
    st.title("PDF 요약하기")
    st.divider()

    pdf = st.file_uploader("PDF 파일 업로드", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        processed_text = process_text(text)
        documents = process_embeds(processed_text)

        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."
        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(question=query, input_documents=docs)
                print(cb)

            st.subheader("요약 결과")
            st.write(response)


if __name__ == "__main__":
    main()

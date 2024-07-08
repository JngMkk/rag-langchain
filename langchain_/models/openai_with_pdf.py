from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

loader = PyPDFLoader(
    "~/workspace/github/rag-langchain/langchain_/data/The_Adventures_of_Tom_Sawyer.pdf"
)
document = loader.load()

# * PDF 6페이지 중 5000글자를 읽어와라
# print(document[5].page_content[:5000])

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(document, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0, model="gpt-4")
# * 검색기
qa = RetrievalQA.from_chain_type(llm, "stuff", retriever=retriever)

query = "마을 무덤에 있떤 남자를 죽인 사람은 누구니?"
result = qa.invoke(query)
print(result)
# {'query': '마을 무덤에 있떤 남자를 죽인 사람은 누구니?', 'result': '마을 무덤에서 남자를 죽인 사람은 Injun Joe입니다.'}

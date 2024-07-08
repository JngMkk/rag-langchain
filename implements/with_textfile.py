from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.chat_models import ChatOpenAI


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


documents = TextLoader("./implements/data/AI.txt").load()
docs = split_docs(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

llm = ChatOpenAI(model="gpt-4")
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

query = "AI란 무엇입니까?"
matching_docs = db.similarity_search(query)
answer = chain.run(question=query, input_documents=matching_docs)
print(answer)
"""
인공 지능 (AI)은 인간이나 동물의 지능이 아닌 기계나 소프트웨어의 지능을 말합니다. 이는 지능적인 기계를 개발하고 연구하는 컴퓨터 과학의 연구 분야입니다. 이러한 기계는 AI라고 불릴 수 있습니다.
AI 기술은 산업, 정부, 과학 등 다양한 분야에서 널리 사용되고 있습니다. 주요한 응용 사례로는 고급 웹 검색 엔진 (예: Google Search), 추천 시스템 (YouTube, Amazon, Netflix 등이 사용), 인간의 음성 이해 (Google Assistant, Siri, Alexa 등), 자율주행 자동차 (예: Waymo), 창조적인 도구 (ChatGPT 및 AI art), 전략 게임의 초인적인 플레이와 분석 (체스와 Go 게임 등)이 있습니다.
AI 연구의 다양한 하위 분야는 특정 목표와 특정 도구의 사용을 중심으로 구성되어 있습니다. AI 연구의 전통적인 목표에는 추론, 지식 표현, 계획, 학습, 자연어 처리, 인식, 로봇 지원 등이 포함됩니다. 일반 지능 (인간이 수행할 수 있는 모든 작업을 완료하는 능력)은 이 분야의 장기 목표 중 하나입니다.
이러한 문제를 해결하기 위해 AI 연구자들은 검색 및 수학적 최적화, 공식 논리, 인공 신경망, 통계, 운영 연구, 경제학 기반의 방법론 등 다양한 문제 해결 기법을 적용하고 통합하였습니다. 또한 AI는 심리학, 언어학, 철학, 신경과학 등 다른 분야에서도 영감을 얻습니다.
"""

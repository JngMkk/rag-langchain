from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4")

prompt1 = PromptTemplate(
    input_variables=["sentence"], template="다음 문장을 한글로 번역해주세요.\n\n{sentence}"
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="translation")

prompt2 = PromptTemplate.from_template("다음 문장을 한 문장으로 요약해주세요.\n\n{translation}")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

all_chains = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["sentence"],
    output_variables=["translation", "summary"],
)

sentence = """
One limitation of LLMs is their lack of contextual information(e.g.,
access to some specific documents or emails). You can combat this by giving
LLMs access to the specific external data.
For this, you first need to load the external data with a document loader.
LangChain provides a variety of loaders for different types of documents
ranging from PDFs and emails to websites and YouTube videos.
"""

print(all_chains.invoke(sentence))
"""
{
    'sentence': '\nOne limitation of LLMs is their lack of contextual information(e.g.,\naccess to some specific documents or emails). You can combat this by giving\nLLMs access to the specific external data.\nFor this, you first need to load the external data with a document loader.\nLangChain provides a variety of loaders for different types of documents\nranging from PDFs and emails to websites and YouTube videos.\n',
    'translation': 'LLM의 한 가지 제한점은 문맥 정보(예: 특정 문서나 이메일에 대한 접근)의 부족입니다. 이를 극복하기 위해 LLM에게 특정 외부 데이터에 대한 접근 권한을 부여할 수 있습니다. 이를 위해 먼저 문서 로더를 사용하여 외부 데이터를 로드해야 합니다. LangChain은 PDF와 이메일부터 웹사이트와 유튜브 비디오에 이르기까지 다양한 유형의 문서에 대한 로더를 제공합니다.',
    'summary': 'LLM의 문맥 정보 부족 문제는 특정 외부 데이터에 접근 권한을 부여하고, LangChain의 문서 로더를 사용해 다양한 유형의 문서를 로드함으로써 극복할 수 있습니다.'
}
"""

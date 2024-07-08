from openai import OpenAI

# * 오픈AI에서 제공하는 임베딩 모델
# * 오픈AI 임베딩은 한국어 지원은 물론, RAG에서 정보 검색과 랭크에 있어서도 우월한 성능을 자랑함.

client = OpenAI(api_key="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")

document = ["제프리 힌튼", "토론토 대학", "사임"]
response = client.embeddings.create(input=document, model="text-embedding-ada-002")

print(response)

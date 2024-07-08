from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# * Gensim은 자연어 처리를 위한 파이썬 라이브러리로, 문서 유사성 분석을 위해 사용됨.

glove_path = "./data/glove.6B/glove.6B.100d.txt"

with open(glove_path, "w") as f:
    f.write("cat 0.5 0.3 0.2\ndog 0.4 0.7 0.8\n")

# * GloVe(Global Vectors for Word Representation)는 단어의 의미를 숫자 벡터로 변환하는 방법 중 하나.
# * 이 방법은 전체 텍스트에서 단어들이 얼마나 자주 함께 나타나는지를 보고 이 정보를 사용해서 각 단어를 벡터로 표현함.

# * Glove File Type to Word2Vec Type
word2vec_output_file = glove_path + ".word2vec"
glove2word2vec(glove_path, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
cat_vector = model["cat"]
print(cat_vector)

from gensim.models import Word2Vec

train_data = [["강아지", "고양이", "두", "마리", "계단", "위", "앉아", "있다"]]

word2vec_model = Word2Vec(sentences=train_data, min_count=1)
word_vector = word2vec_model.wv["강아지"]
print(word_vector)

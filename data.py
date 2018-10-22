import csv
import gensim
import pickle
from gensim.models import Word2Vec

# sentences = []
# with open('training_data.csv', 'r') as csvFile:
#     reader = csv.reader(csvFile)
#     for row in reader:
#         #print(row[1])
#         sentences.append(row[1])

# csvFile.close()

# print(len(sentences))

# model = gensim.models.Word2Vec(
#         sentences,
#         size=150,
#         window=10,
#         min_count=2,
#         workers=10)
# model.train(sentences, total_examples=len(sentences), epochs=10)
# print("Done")

#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#model.init_sims(replace=True)
#model.save('google_word')

#later load the model
loaded_model = Word2Vec.load('google_word')

w1 = "friend"

print(loaded_model.wv.most_similar(positive = w1))

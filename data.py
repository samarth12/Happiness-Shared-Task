import csv
import gensim
import pickle
from gensim.models import Word2Vec

#Load the dataset sentences
def load_data(file):
    sentences = []
    with open(file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            #print(row[1])
            sentences.append(row[1])

    csvFile.close()
    return sentences

#Train word2vec model for input sentences
def train_wv(data):
    model = gensim.models.Word2Vec(
            data,
            size=150,
            window=10,
            min_count=2,
            workers=10)
    model.train(data, total_examples=len(data), epochs=10)
    print("Done")
    return model


#Load the Google News Word2Vec
def load_google_wv(file):
    model =  gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)  
    return model


def main():


    sentences = load_data('training_data.csv')
    #print(sentences)
    loaded_model = load_google_wv('GoogleNews-vectors-negative300.bin')
    #print(loaded_model.vocab)

    w1 = "friend"
    w2 = "pal"
    #Print most similar words
    print("Top 10 most similar words to w1:",loaded_model.wv.most_similar(positive = w1, topn = 10))
    
    #Print similarity between two word
    print("Similarity:", loaded_model.wv.similarity(w1, w2))


main()

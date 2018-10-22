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
    model =  Word2Vec.load_word2vec_format(file, binary=True)  
    return model


def main():


    load_data('training_data.csv')
    loaded_model = load_google_wv('GoogleNews-vectors-negative300.bin')

    w1 = "friend"
    print(loaded_model.wv.most_similar(positive = w1))


main()

import spacy
from sklearn import svm
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re

def getTrainSets():
    with open('dataset/hate/train_text.txt', 'r', encoding='utf-8') as infile:
        train_x = [line.strip() for line in infile]
    with open('dataset/hate/train_labels.txt', 'r') as infile:
        train_y = [int(line) for line in infile]
    return train_x, train_y

def trainWordVector(nlp, train_x, train_y):
    #the code below deletes stopwords but did not show a difference in the accuracy of predictions 
    pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
    train_x_processed = []
    for line in train_x:
        train_x_processed.append(pattern.sub('', line))
    
    #the tokenizer currently does not work because it creates a list of list and that cannot be fed to nlp
    tokenizer = TweetTokenizer()
    train_x_tokenized = [tokenizer.tokenize(line) for line in train_x]
    print(train_x_tokenized[:10])

    temp = [nlp(text) for text in train_x_tokenized]
    train_x_vectors = [x.vector for x in temp]

    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(train_x_vectors, train_y)
    return clf_svm

def runValidation(nlp, clf_svm):
    with open('dataset/hate/val_text.txt', 'r', encoding= 'utf-8') as infile:
        val_x = [line.strip() for line in infile]

    with open('dataset/hate/val_labels.txt', 'r') as infile:
        val_y = [int(line) for line in infile]

    val_temp = [nlp(text) for text in val_x]
    val_x_vectors = [x.vector for x in val_temp]

    return clf_svm.predict(val_x_vectors), val_y

def getAccuracy(predictions, val_y):
    correct = 0
    for a, b in zip(predictions, val_y):
        if a + b == 2:
            correct += 1
        elif a + b == 0:
            correct += 1
    print(correct/len(val_y))

def main():
    nlp = spacy.load('en_core_web_lg')
    train_x, train_y = getTrainSets()
    clf_svm = trainWordVector(nlp, train_x, train_y)
    nlp = spacy.load('en_core_web_lg')
    predictions, val_y = runValidation(nlp, clf_svm)
    getAccuracy(predictions, val_y)

if __name__ == '__main__':
    main()

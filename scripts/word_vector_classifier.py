import spacy
from sklearn import svm
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score

#python -m spacy download en_core_web_sm needs to be run in the terminal before running this
#python -m nltk.downloader stopwords needs to be run in the terminal before running this

#THINGS TO LOOK INTO
#sklearn.feature_extraction.text.TfidfTransformer
#huggingface transformers and recurring neural network analysis


class getVectorClassifier():
    def __init__(self, train_text_path, train_labels_path):
        """ initialize the self variables as well as get the training sets and the classifier"""
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.nlp = spacy.load('en_core_web_lg')
        self.train_x, self.train_y = self.getTrainSets()
        self.clf_svm = self.trainWordVector()

    def getTrainSets(self):
        """ get both the train text and the labels for the training set """
        with open(self.train_text_path, 'r', encoding='utf-8') as infile:
            train_x = [line.strip() for line in infile]

        with open(self.train_labels_path, 'r') as infile:
            train_y = [int(line) for line in infile]

        return train_x, train_y

    def getStopWords(self):
        return stopwords.words('english')
        #against, 

    def trainWordVector(self):
        """ uses the training data, 
        processes it by deleting stopwords and tokenizing as well as vectorizing the sentence, 
        returns a classifier """
        stop_words = self.getStopWords()
        pattern = re.compile(r'\b(' +r'|'.join(stop_words) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(pattern.sub('', line))
        
        #extra tokenizer that is not necessary bc spacy tokenizes automatically 
        # tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        # train_x_tokenized = [tokenizer.tokenize(line) for line in self.train_x]
        train_x_corrected = []
        c = 0
        for tweet in self.train_x:
            tw = TextBlob(tweet)
            c += 1
            if (c % 10 == 0): print(c)
            
            train_x_corrected.append(str(tw.correct()))

        temp = [self.nlp(text) for text in train_x_corrected] 
        train_x_vectors = [x.vector for x in temp]

        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm 

    def returnClassifier(self):
        return self.clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        stopwords_processed = self.getStopWords()
        pattern = re.compile(r'\b(' +r'|'.join(stopwords_processed) +r')\b\s')
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(pattern.sub('', line))

        val_temp = [self.nlp(text) for text in val_x]
        val_x_vectors = [x.vector for x in val_temp]
        predictions = self.clf_svm.predict(val_x_vectors)

        correct = 0
        for a, b in zip(predictions, val_y):
            if a == b:
                correct += 1

        acc = correct/len(val_y)

        return f1_score(predictions, val_y, average='macro')

class bagOfWordsClassifier():
    """ creates a bag of words classifier using the train data """
    def __init__(self, train_text_path, train_labels_path):
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.train_x, self.train_y = self.getTrainSets()
        self.clf_svm = self.initializeBOWClassifier()
    
    def getTrainSets(self):
        """ get both the train text and the labels for the training set """
        with open(self.train_text_path, 'r', encoding='utf-8') as infile:
            train_x = [line.strip() for line in infile]

        with open(self.train_labels_path, 'r') as infile:
            train_y = [int(line) for line in infile]

        return train_x, train_y

    def initializeBOWClassifier(self):
        pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(pattern.sub('', line))
 

        self.vectorizer = CountVectorizer(binary=True)
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)

        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        val_x_vectors = self.vectorizer.transform(val_x)
        predictions = self.clf_svm.predict(val_x_vectors)

        correct = 0
        for a, b in zip(predictions, val_y):
            if a == b:
                correct += 1

        return f1_score(predictions, val_y)

    def returnClassifier(self):
        return self.clf_svm


class getHarmfulness():
    """ processes a single tweet and gets its harmfullness as well as the general sentiment of the tweet """
    def __init__(self, tweet, clf_svm, nlp):
        self.tweet = tweet
        self.clf_svm = clf_svm
        self.tweet_vectors = self.processTweet()
        self.nlp = nlp

    def processTweet(self):
        #run spell check on the tweet 
        stop_words = stopwords.words('english')
        tweet_stripped = []
        for word in self.tweet:
            if word not in stop_words:
                tweet_stripped.append(word)
        val_temp = [self.nlp(text) for text in tweet_stripped]
        tweet_vectors = [x.vector for x in val_temp]
        return tweet_vectors

    def harmfulness(self):
        return self.clf_svm.predict(self.tweet_vectors)

    def getSentiment(self):
        tb_tweet = TextBlob(self.tweet)
        return tb_tweet.sentiment


def main():
    #create nlp model
    nlp = spacy.load('en_core_web_lg')

    #paths for the training and validation data
    train_text_path = 'dataset/hate/train_text.txt'
    train_labels_path = 'dataset/hate/train_labels.txt'
    val_text_path = 'dataset/hate/val_text.txt'
    val_labels_path = 'dataset/hate/val_labels.txt'

    #create a classifier using the vector classifier class
    vec_classifier = getVectorClassifier(train_text_path, train_labels_path)
    clf_svm = vec_classifier.returnClassifier()
    print(vec_classifier.getClassifierAccuracy(val_text_path, val_labels_path))

    #create a classifier using the bag of words class
    # bag_classifier = bagOfWordsClassifier(train_text_path, train_labels_path)
    # print(bag_classifier.getClassifierAccuracy(val_text_path, val_labels_path))

    # tweet =  input() #change this so that there can be multiple inputs
    # tweetObj = getHarmfulness(tweet, clf_svm, nlp)
    # harmfulness = tweetObj.harmfulness()
    # print(harmfulness)

if __name__ == '__main__':
    main()




#general imports
import numpy as np
import re 

#sklearn imports
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import f1_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 

#nltk imports
from nltk.corpus import stopwords 
from nltk.tokenize import TweetTokenizer 

#additional imports used for natural language processing
import spacy 

#IN ORDER TO RUN THIS PYTHON SCRIPT:
#both the stopwords and the spacy trained pipeline for english need to be installed on the environment you're trying to run the code in
#the commands for that are
#python -m spacy download en_core_web_lg
#python -m nltk.downloader stopwords

class WithStopVectorClassifier:
    """ The purpose of this class is to train a classifier using the spacy pipeline and a given train set,
    It does not remove stopwords. """
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

    def trainWordVector(self):
        """ processes the training data by using the spacy pipeline, 
        then turns it into a support vector classifier """

        temp = [self.nlp(text) for text in self.train_x] #works as a tokenizer, lemmatizer etc. all in one
        train_x_vectors = [x.vector for x in temp] #turns the spacy objects from the step before into vectors

        #use the vectors to train a support vector classifier
        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm 

    def returnClassifier(self):
        """ return the trained classifier """
        return self.clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score of the model """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]
        
        val_temp = [self.nlp(text) for text in val_x]
        val_x_vectors = [x.vector for x in val_temp]
        predictions = self.clf_svm.predict(val_x_vectors)

        return f1_score(predictions, val_y, average='macro')

class NoStopVectorClassifier:
    """ The purpose of this class is to train a classifier using the spacy pipeline and a given train set,
    It removes stopwords. """
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

    def trainWordVector(self):
        """ uses the training data, 
        processes it by deleting stopwords and tokenizing as well as vectorizing the sentence, 
        returns a classifier """
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(pattern.sub('', line))

        temp = [self.nlp(text) for text in train_x_processed] #works as a tokenizer, lemmatizer etc. all in one
        train_x_vectors = [x.vector for x in temp] #turns the spacy objects from the step before into vectors

        #use the vectors to train a support vector classifier
        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm 

    def returnClassifier(self):
        """ return the trained classifier """
        return self.clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score of the model """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        val_temp = [self.nlp(text) for text in val_x_processed]
        val_x_vectors = [x.vector for x in val_temp]
        predictions = self.clf_svm.predict(val_x_vectors)

        return f1_score(predictions, val_y, average='macro')

class BagOfWordsClassifier:
    """ The purpose of a class is to train a classifier using the bag of words method and a given train set,
    it utilises the TweetTokenizer from nltk and there is no stopword removal """
    def __init__(self, train_text_path, train_labels_path):
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.train_x, self.train_y = self.getTrainSets()
        self.clf_svm = self.initialiseBOWClassifier()
    
    def getTrainSets(self):
        """ get both the train text and the labels for the training set """
        with open(self.train_text_path, 'r', encoding='utf-8') as infile:
            train_x = [line.strip() for line in infile]

        with open(self.train_labels_path, 'r') as infile:
            train_y = [int(line) for line in infile]

        return train_x, train_y

    def initialiseBOWClassifier(self):
        """ initialise and return a bag of words classifier """
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))
 
        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        self.vectorizer = CountVectorizer( tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)

        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score for the classifier """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        #remove stopwords
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))
        
        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.clf_svm.predict(val_x_vectors)

        return f1_score(predictions, val_y, average = 'macro')

    def returnClassifier(self):
        return self.clf_svm

class RandomForest:
    """ The purpose of this class is to train a random forest classifier with a train data set,
    It utilises the TweetTokenizer from nltk and there is stopword removal. """
    def __init__(self, train_text_path, train_labels_path):
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.train_x, self.train_y = self.getTrainSets()
        self.rf_clf = self.randomForest()

    def getTrainSets(self):
        """ get both the train text and the labels for the training set """
        with open(self.train_text_path, 'r', encoding='utf-8') as infile:
            train_x = [line.strip() for line in infile]

        with open(self.train_labels_path, 'r') as infile:
            train_y = [int(line) for line in infile]

        return train_x, train_y 

    def randomForest(self):
        """ initialize and return a random forest classifier """
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))

        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        self.vectorizer = CountVectorizer(tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)
        rf_clf = RandomForestClassifier(criterion='entropy')
        rf_clf.fit(train_x_vectors, self.train_y)
        return rf_clf
    
    def getAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score of the classifier """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.rf_clf.predict(val_x_vectors)

        return f1_score(predictions, val_y, average = 'macro')

class DecisionTree:
    def __init__(self, train_text_path, train_labels_path):
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.train_x, self.train_y = self.getTrainSets()
        self.dt_clf = self.decisionTree()

    def getTrainSets(self):
        """ get both the train text and the labels for the training set """
        with open(self.train_text_path, 'r', encoding='utf-8') as infile:
            train_x = [line.strip() for line in infile]

        with open(self.train_labels_path, 'r') as infile:
            train_y = [int(line) for line in infile]

        return train_x, train_y 

    def decisionTree(self):
        pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(pattern.sub('', line))

        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        self.vectorizer = CountVectorizer(tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(train_x_vectors, self.train_y)
        return dt_clf

    def getAccuracy(self, val_x_path, val_y_path):
        with open(val_x_path, 'r', encoding = 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.dt_clf.predict(val_x_vectors) 

        return f1_score(predictions, val_y, average = 'macro')
        
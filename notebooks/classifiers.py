#general imports
import numpy as np
import re 

#sklearn imports
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 

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
    def __init__(self, train_text_path, train_labels_path, already=False):
        """ initialize the self variables as well as get the training sets and the classifier"""
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.nlp = spacy.load('en_core_web_lg')  #load in the pre-trained pipeline from spacy
        self.train_x, self.train_y = self.getTrainSets(already)
        self.clf_svm = self.trainWordVector()

    def getTrainSets(self, already=False):
        """ load in the train data set, train_x has the text and train_y the labels """
        if not already:
            with open(self.train_text_path, 'r', encoding='utf-8') as infile:
                train_x = [line.strip() for line in infile]

            with open(self.train_labels_path, 'r') as infile:
                train_y = [int(line) for line in infile]
        
        else:
            train_x = self.train_text_path
            train_y = self.train_labels_path
        return train_x, train_y

    def trainWordVector(self):
        """ processes the training data by using the spacy pipeline, 
        then turns it into a fully trained support vector classifier """

        temp = [self.nlp(tweet) for tweet in self.train_x] #works as a tokenizer, lemmatizer etc. all in one
        train_x_vectors = [x.vector for x in temp] #turns the spacy objects from the step before into vectors

        #use the vectors to train a support vector classifier
        clf_svm = svm.SVC(kernel='linear') #we use a linear kernel bc we found that it is the best one
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm 

    def returnClassifier(self):
        """ return the trained classifier """
        return self.clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path, already=False):
        """ use the validation set to get the f1 score, balanced accuracy, recall and precision of the model """
        if not already:
            with open(val_x_path, 'r', encoding= 'utf-8') as infile:
                val_x = [line.strip() for line in infile]

            with open(val_y_path, 'r') as infile:
                val_y = [int(line) for line in infile]
        else:
            val_x = val_x_path
            val_y = val_y_path
        
        val_temp = [self.nlp(text) for text in val_x]
        val_x_vectors = [x.vector for x in val_temp]
        predictions = self.clf_svm.predict(val_x_vectors)

        f1 = f1_score( val_y, predictions, average='macro')
        acc = accuracy_score(val_y, predictions)
        prec = precision_score(val_y, predictions, average = 'macro')
        rec = recall_score(val_y, predictions, average = 'macro')

        return f1, acc, prec, rec 

class NoStopVectorClassifier:
    """ The purpose of this class is to train a classifier using the spacy pipeline and a given train set,
    It removes stopwords. """
    def __init__(self, train_text_path, train_labels_path):
        """ initialize the self variables as well as get the training sets and the classifier"""
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.nlp = spacy.load('en_core_web_lg')  #load in the pre-trained pipeline from spacy
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
        """ uses the training data, processes it by deleting stopwords, 
        returns a trained classifier """

        #create a new list with processed tweets that do not have stopwords
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))

        temp = [self.nlp(text) for text in train_x_processed] #works as a tokenizer, lemmatizer etc. all in one
        train_x_vectors = [x.vector for x in temp] #turns the spacy objects from the step before into vectors

        #use the vectors to train a support vector classifier
        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm 

        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
        clf.fit(train_x_vectors, self.train_y)
        Pipeline(steps=[('standardscaler', StandardScaler()),
                ('sgdclassifier', SGDClassifier())])
        print(clf.predict([[-0.8, -1]]))

    def returnClassifier(self):
        """ return the trained classifier """
        return self.clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score, balanced accuracy, recall and precision of the model """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        #deleting stopwords
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        #turning the validation set into vectors
        val_temp = [self.nlp(text) for text in val_x_processed]
        val_x_vectors = [x.vector for x in val_temp]

        #getting the predicted labels for the validation set
        predictions = self.clf_svm.predict(val_x_vectors)

        f1 = f1_score( val_y, predictions, average='macro')
        acc = accuracy_score(val_y, predictions)
        prec = precision_score(val_y, predictions, average = 'macro')
        rec = recall_score(val_y, predictions, average = 'macro')

        return f1, acc, prec, rec 

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
        """ initialise, train and return a bag of words classifier """
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))
 
        #initialize tweet tokenizer to pass into the count vectorizer 
        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        self.vectorizer = CountVectorizer(tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data

        #the training set needs to be fit transformed before it is passed into the classifier
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)

        clf_svm = svm.SVC(kernel='linear')
        clf_svm.fit(train_x_vectors, self.train_y)
        return clf_svm

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score, balanced accuracy, recall and precision of the model """
        with open(val_x_path, 'r', encoding= 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        #remove stopwords
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))
        
        #transforming the validation data and get the predicted labels
        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.clf_svm.predict(val_x_vectors)

        f1 = f1_score( val_y, predictions, average='macro')
        acc = accuracy_score(val_y, predictions)
        prec = precision_score(val_y, predictions, average = 'macro')
        rec = recall_score(val_y, predictions, average = 'macro')

        return f1, acc, prec, rec 

    def returnClassifier(self):
        """ returns the fully trained classifier """
        return self.clf_svm

class RandomForest:
    """ The purpose of this class is to train a random forest classifier with a train data set,
    It utilises the TweetTokenizer from nltk and there is stopword removal. """
    def __init__(self, train_text_path, train_labels_path, already=False):
        self.train_text_path = train_text_path
        self.train_labels_path = train_labels_path
        self.train_x, self.train_y = self.getTrainSets(already)
        self.rf_clf = self.randomForest()

    def getTrainSets(self, already):
        """ get both the train text and the labels for the training set """
        if not already:
            with open(self.train_text_path, 'r', encoding='utf-8') as infile:
                train_x = [line.strip() for line in infile]

            with open(self.train_labels_path, 'r') as infile:
                train_y = [int(line) for line in infile]
        else:
            train_x = self.train_text_path
            train_y = self.train_labels_path

        return train_x, train_y 

    def randomForest(self):
        """ initialize, train and return a random forest classifier """

        #delete stopwords
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))

        #initialize an instance of the tweet tokenizer to pass into the countvectorizer
        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        #fit transform the training data in order to pass it into the classifier
        self.vectorizer = CountVectorizer(tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)
        rf_clf = RandomForestClassifier(criterion='entropy')
        rf_clf.fit(train_x_vectors, self.train_y)
        return rf_clf
    
    def getClassifierAccuracy(self, val_x_path, val_y_path, already=False):
        """ use the validation set to get the f1 score, balanced accuracy, recall and precision of the model """
        if not already:
            with open(val_x_path, 'r', encoding= 'utf-8') as infile:
                val_x = [line.strip() for line in infile]

            with open(val_y_path, 'r') as infile:
                val_y = [int(line) for line in infile]
        else:
            val_x = val_x_path
            val_y = val_y_path
            
        #delete stopwords
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        #transform the validation data and get the predicted labels
        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.rf_clf.predict(val_x_vectors)

        f1 = f1_score( val_y, predictions, average='macro')
        acc = accuracy_score(val_y, predictions)
        prec = precision_score(val_y, predictions, average = 'macro')
        rec = recall_score(val_y, predictions, average = 'macro')

        return f1, acc, prec, rec 

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
        """ initialize, train and return a decision tree classifier """

        #delete stopwords
        self.pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english')) +r')\b\s')
        train_x_processed = []
        for line in self.train_x:
            train_x_processed.append(self.pattern.sub('', line))

        #initialize tweet tokenizer to pass into the vectorizer
        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)

        self.vectorizer = CountVectorizer(tokenizer=lambda text: tw_tokenizer.tokenize(text)) #deleted binary=True when running with abortion data
        
        #fit transform the training data using the vectorizer
        train_x_vectors = self.vectorizer.fit_transform(train_x_processed)

        #initialize and train decision tree classifier
        dt_clf = DecisionTreeClassifier()
        dt_clf.fit(train_x_vectors, self.train_y)
        return dt_clf

    def getClassifierAccuracy(self, val_x_path, val_y_path):
        """ use the validation set to get the f1 score, balanced accuracy, recall and precision of the model """
        with open(val_x_path, 'r', encoding = 'utf-8') as infile:
            val_x = [line.strip() for line in infile]

        with open(val_y_path, 'r') as infile:
            val_y = [int(line) for line in infile]

        #delete stopwords
        val_x_processed = []
        for line in val_x:
            val_x_processed.append(self.pattern.sub('', line))

        #transform validation set and get predicted labels
        val_x_vectors = self.vectorizer.transform(val_x_processed)
        predictions = self.dt_clf.predict(val_x_vectors) 

        f1 = f1_score( val_y, predictions, average='macro')
        acc = accuracy_score(val_y, predictions)
        prec = precision_score(val_y, predictions, average = 'macro')
        rec = recall_score(val_y, predictions, average = 'macro')

        return f1, acc, prec, rec 
        
#!/usr/bin/python

""" Author: Peter Swanson
            pswanson@ucdavis.edu
    Description: Naive Bayesian Classifier
    Version: Python 2.7 """

import pickle
from textblob.classifiers import NaiveBayesClassifier
from random import shuffle

class Bayes_Classifier(object):
    ''' Classifier and related methods '''

    # Name of file to store classifier in
    file = "classifier.obj"

    def __init__(self, load=False):
        # Can be set to load classifier with saved data
        if load:
            self.load()
        else:
            self.classifier = None
            self.is_trained = False
            self.data = None

    def clear(self):
        ''' Clears classifier data '''
        
        self.classifier = None
        self.is_trained = False
        self.data = None

    def update_and_train_data(self, new_data):
        ''' Update the classifier with new training data '''

        # Create new classifier with data
        self.classifier = NaiveBayesClassifier(new_data)
        # Set new data
        self.data = new_data

    def add_data_and_train(self, new_data):
        ''' Add training data to the classifier '''

        if not self.classifier or not self.data:
            print "Error - No data to update."
            return -1

        # Get old data
        up_data = self.data

        # Mesh with new data
        for data in new_data:
            up_data.append(data)

        # Update and train
        self.update_and_train_data(up_data)

    def save(self):
        ''' Saves the classifier '''

        if not self.classifier:
            print "Error - No classifier to save."
            return -1

        # Dump to classifier.obj
        pickle.dump((self.classifier, self.is_trained, self.data), \
                      open(self.file,'wb'))

    def load(self):
        ''' Loads a classifier '''

        # Try to load classifier.obj
        try:
            self.classifier, self.is_trained, self.data = pickle.load(open(self.file,'rb'))
        except IOError:
            print "Error - Could not find a classifier to load."
            exit(1)

    def train_model(self, data_set):
        ''' Train the model on parsed data '''

        if not self.is_trained:
            if len(data_set) > 0:
                self.classifier = NaiveBayesClassifier(data_set)
                self.is_trained = True
                self.data = data_set
                return
            else:
                print "Error - Training data set empty."
        else:
            print "Error - Model already trained."

        return -1

    def get_model_details(self):
        ''' Get details about a trained model '''

        if not self.is_trained:
            print "Error - Model not trained."
            return -1
    
        self.classifier.show_informative_features()

    def test_model(self, data_set):
        ''' Test the model on parsed data '''

        if not self.is_trained:
            print "Model not trained"
            return -1
        
        if len(data_set) > 0:
            if isinstance(data_set[0], tuple):
                # Classify each sentence from tuple
                for data in data_set:
                    print '\"' + data[0] + '\"' + " evaluates to " + self.classifier.classify(data[0]) + "."
            
            elif isinstance(data_set[0], str):
                # Classify each sentence
                for data in data_set:
                    print '\"' + data + '\"' + " evaluates to " + self.classifier.classify(data) + "."

            else:
                # Invalid data format
                print "Error - Test data not a list of tuples or strings."
                return -1

            try:
                print "\nClassified with " + str(float(self.classifier.accuracy(data_set)) * 100) + "% accuracy." 
            except ValueError:
                print "\nTest data not labeled. Could not determine accuracy."

        else:
            print "Error - Test data set is empty."
            return -1

    def split_60_40(self, data_set):
        ''' Shuffle and split data into training and testing set '''

        shuffle(data_set)

        length = len(data_set)

        split_index = int(length * .6)

        return(data_set[:split_index:], data_set[split_index::])
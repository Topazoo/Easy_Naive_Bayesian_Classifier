#!/usr/bin/python

""" Author: Peter Swanson
            pswanson@ucdavis.edu
    Description: Driver to run classifier examples
    Version: Python 2.7 """

from Bayes_Classifier import Bayes_Classifier
import sys

# Data to teach the classifier to identify positive and negative sentences
training_data = [
    ('this is awful bad', 'neg'),
    ('awful day', 'neg'),
    ('bad week', 'neg'),
    ('that awful dog', 'neg'),
    ('my bad cat', 'neg'),
    ('what a good time', 'pos'),
    ('have a great summer', 'pos'),
    ('is it good or great', 'pos'),
    ('I had a good one', 'pos'),
    ('great excitement', 'pos'),
]

# Data to test the classifier with
testing_data = [
    ('An awful encounter', 'neg'),
    ('How great', 'pos'),
    ('Thats a bad one', 'neg'),
    ('terrible winter', 'neg'),
    ('One amazing summer', 'pos'),
    ('A terrible, bad friday', 'neg')
]

# Additional data needed to correctly classify all sentences in test set
new_data = [
    ('this is amazing bad', 'pos'),
    ('wonderful day', 'pos'),
    ('wonderful week', 'pos'),
    ('that amazing dog', 'pos'),
    ('my wonderful cat', 'pos'),
    ('what a terrible time', 'neg'),
    ('have a horrible summer', 'neg'),
    ('is it terrible or horrible', 'neg'),
    ('I had a horrible one', 'neg'),
    ('terrible excitement', 'neg'),
]

def add_data_demo():
    ''' Add data to the classifier's training set and reclassify the testing data '''
    
    # Create a classifier
    cl = Bayes_Classifier()

    # Train the classifier on the training data
    cl.train_model(training_data)

    # Add and retrain the model
    cl.add_data_and_train(new_data)

    # Test the classifer
    cl.test_model(testing_data)

def novel_classification_demo():
    ''' Classify the sentences in the test set based on the training set'''

    # Create a classifier
    cl = Bayes_Classifier()

    # Train the classifier
    cl.train_model(training_data)

    # Test the classifer
    cl.test_model(testing_data)

def classification_accuracy_demo():
    ''' Split full training set 60/40 to create a smaller test set, then determine classifier accuracy on test set'''

    # Create a classifier
    cl = Bayes_Classifier()

    # Create full data set with all data labeled
    all_data = training_data + new_data

    # Split into training and testing sets
    train_set, test_set = cl.split_60_40(all_data)

    # Train the classifier
    cl.train_model(train_set)

    # Test the classifer
    cl.test_model(test_set)


# Main driver to select the demo to run
# Default is classifying a novel test set with incomplete information to determine accuracy

if __name__ == "__main__":
    to_run = "novel"

    if len(sys.argv) > 1:
        to_run = str(sys.argv[1])

    if to_run == "novel":
        novel_classification_demo()
    elif to_run == "all":
        add_data_demo()
    elif to_run == "accuracy":
        classification_accuracy_demo()
    else:
        print "Demo " + to_run + " not found!"
        exit(1)
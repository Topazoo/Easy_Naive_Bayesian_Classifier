# Easy Naive Bayesian Classifier
### Author: Peter Swanson
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 2.7](https://img.shields.io/badge/Python-2.7-brightgreen.svg)](https://www.python.org/downloads/release/python-2714/)
[![nltk 3.3](https://img.shields.io/badge/nltk-3.3-brightgreen.svg)](https://pypi.org/project/nltk/)
[![textblob 0.15.1](https://img.shields.io/badge/textblob-0.15.1-brightgreen.svg)](https://pypi.org/project/textblob/)

## Background:
<b>This application greatly simplifies training and using a Naive Bayesian Classifier to classify text.</b>

This repository contains the Python code for training and running the classifier, as well as a sample driver with three demos.

## Running the Application:
### Installing Dependencies:
Ensure the following are installed on the machine you are running the application on:
- Python 2.7 with pip
- virtualenv for Python 2.7

Create a virtualenv and install the requirements from <i>requirements.txt</i> with pip
```
$ pip install -r "requirements.txt"
``` 

### Running the Driver:
The demo driver can be run with Python:
```
$ python driver.py
```

By default, the driver runs the novel classification demo. In this demo it uses an incomplete training set
to classify data in a testing set. It then displays the testing set data with the classification the classifier assigned, and the classifier's overall
accuracy.

There are two more demos included:

```
$ python driver.py all
```

This demo classifies testing data with a complete training set, resulting in 100% accuracy.

```
$ python driver.py accuracy
```

This demo splits the training data into a random, 60/40 training and testing set each time it is run. It then displays the testing set data with the classification the classifier assigned, and the classifier's overall
accuracy.

### Classifying Text:
The Classifier can be imported and used with other Python applications or run from the console.
Classifiers may optionally be saved and loaded with any loaded data intact. Classifiers are saved with the name
"Classifier.obj."

#### Instantiation:
```
>>> from Bayes_Classifier import Bayes_Classifier
>>> cl = Bayes_Classifier(load=False)
```

#### Training the Classifier:
The classifier can be trained using the <i>train_model()</i> method.
```
>>> training_data = [('A bad day', 'neg'), ('One great week', 'pos')]
>>> cl.train_model(training_data)
```

#### Running the Classifier:
Once trained, the Classifier can be run on a data set using the <i>test_model()</i> method. Testing data may be optionally labeled to determine the accuracy of the classifier.
If test data is unlabeled, the classifier will be unable to determine accuracy. 
```
>>> test_data = [('What great food', 'pos'), ('That bad dog', 'neg')]
>>> cl.test_model(test_data)
"What great food" evaluates to pos.
"That bad dog" evaluates to neg.

Classified with 100.0% accuracy
```

#### Formatting Data:
Training data must be in the form of a list of tuples. Each tuple contains a string and its classification.
e.g.
```
training_data = [
    ('this is awful', 'neg'),
    ('awful day', 'neg'),
    ('bad week', 'neg'),
    ('what a good time', 'pos'),
    ('I had a good one', 'pos'),
    ('great excitement', 'pos'),
]
```

Testing data must mimic training data to receive an accuracy score. However, the classifier accepts testing data in the form of a list
of test strings instead of (string, classification) tuples.
e.g.
```
test_data = [
    ('An awful encounter', 'neg'),
    ('How great', 'pos'),
]

OR

test_data = [
    'An awful encounter',
    'How great',
]
```

## Files
- <i>driver.py</i> - Demo script for the classifier.
- <i>Bayes_Classifier.py</i> - Class for classifying text based on a training set.
    
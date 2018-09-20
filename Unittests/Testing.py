import unittest, os
from Bayes_Classifier import Bayes_Classifier

class LoadTest(unittest.TestCase):
    def test_bad_load(self):
        with self.assertRaises(SystemExit) as cm:
            Bayes_Classifier(load=True)

        self.assertEqual(cm.exception.code, 1)

    def test_save_and_load(self):
        data = [('test', 'pos')]
        c = Bayes_Classifier()
        c.train_model(data)
        c.save()
        c.clear()

        d = Bayes_Classifier(load=True)
        self.assertEqual(d.data, data)

        os.remove('classifier.obj')

class RunTest(unittest.TestCase):
    def test_run(self):
        c = Bayes_Classifier()
        training_data = [('A bad day', 'neg'), ('One great week', 'pos')]
        test_data = [('What great food', 'pos'), ('That bad dog', 'neg')]
        c.train_model(training_data)

        self.assertEqual(c.classifier.accuracy(test_data), 1.0)

if __name__ == '__main__':
    unittest.main()

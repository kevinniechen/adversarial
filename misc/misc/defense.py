from art.classifiers import KerasClassifier
import numpy as np

class Defense():
    """ Provides read-only access to attack model variables. """
    def __init__(self,
                 x_train,
                 y_train,
                 adv_train,
                 x_test,
                 y_test,
                 adv_test,
                 classifier,
                 model):
        self.x_train = x_train
        self.y_train = y_train
        self.adv_train = adv_train
        self.x_test = x_test
        self.y_test = y_test
        self.adv_test = adv_test

    def nomutation(func):
        """ Ensure that class variables did not change. """
        def function_wrapper(x):
            cache = vars(self)
            res = func(x)
            assert(cache == vars(self))
            return res
        return function_wrapper

    @nomutation
    def adversarial_training(self):
        # Data augmentation: expand the training set with the adversarial samples
        x_train = np.append(self.x_train, self.adv_train, axis=0)
        y_train = np.append(self.y_train, self.y_train, axis=0)
        
        # Retrain the CNN on the extended dataset
        classifier = KerasClassifier((min_, max_), model=model)
        classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)


        
        with open(out_file, 'a+') as f:
            preds = np.argmax(classifier.predict(x_train), axis=1)
            acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
            print("TRAIN: %.2f%% \n" % (acc * 100), file=f)
            
            preds = np.argmax(classifier.predict(self.adv_train), axis=1)
            acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
            print("TRAIN-ADVERSARY: %.2f%% \n" % (acc * 100), file=f)
            
            preds = np.argmax(classifier.predict(x_test), axis=1)
            acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
            print("TEST: %.2f%% \n" % (acc * 100), file=f)
            
            preds = np.argmax(classifier.predict(self.adv_test), axis=1)
            acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
            print('TEST-ADVERSARY: %.2f%% \n' % (acc * 100), file=f)

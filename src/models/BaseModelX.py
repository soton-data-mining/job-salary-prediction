import pickle

import os.path
import pandas as pd
import sklearn

from models.FeatureObject import FeatureObject


class BaseModelX(object):
    TRAIN_NORMALIZED_LOCATION_FILE_NAME = '../data/train_normalised_location.csv'
    TRAIN_DATA_CSV_FILE_NAME = '../data/Train_rev1.csv'
    CLEANED_FILE_NAME = '../data/cleaned_data.pickle'

    def __init__(self, train_size=0.75, test_size=None, features=None):
        """
        :param train_size: can be either a float or int
         - float: ratio of how much is training/test data
         - int: for total size
        :param test_size: see train_size - allows you to work on a smaller data set
         i.e. 100 train, 50 test for debugging or during development
         defaults to None which equals 1-train_size
        """
        print('Initialized {}'.format(self.__class__.__name__))

        if features:
            print('Pre-processed data exists, reading from parameter')
            self.features = features
        elif os.path.exists(BaseModelX.CLEANED_FILE_NAME):
            print('Pre-processed data exists, reading from the file')
            self.features = self.load_cleaned_data()
            # csv export is not needed anymore, since we use pickle
            # but if we want to look at the data, we can export it
            # self.features.export_csv('../data/sample_export.csv')
            print('Data read complete')
        else:
            print('Pre-processed data doesn\'t exist, preprocessing data first')
            print('This operation will take a while')
            data, normalized_location_data = self.load_all_data()

            self.features = FeatureObject()
            self.features.preprocess_data(data, normalized_location_data)
            pickle.dump(self.features, open(BaseModelX.CLEANED_FILE_NAME, 'wb'))
            print('Raw data read complete')

        print('Splitting train and test')
        self.features.split_training_test(train_size, test_size)
        print('Train test split complete \n')

    @staticmethod
    def export_prediction(prediction_to_write, file_name):
        print('Exporting prediction to ../predictions/' + file_name)
        if not (os.path.exists("../predictions")):
            os.makedirs("../predictions")
        f = open('../predictions/' + file_name + '.csv', 'a')
        for index, item in enumerate(prediction_to_write):
            f.write(str(item))
            if index < len(prediction_to_write) - 1:
                f.write('\n')
        f.close()
        print('Exporting complete \n')

    @staticmethod
    def load_all_data():
        """
        loads all data into the static set
        """
        data = pd.read_csv(BaseModelX.TRAIN_DATA_CSV_FILE_NAME)
        normalized_location_data = pd.read_csv(
            BaseModelX.TRAIN_NORMALIZED_LOCATION_FILE_NAME)
        return data, normalized_location_data

    @staticmethod
    def load_cleaned_data():
        features = pickle.load(open(BaseModelX.CLEANED_FILE_NAME, 'rb'))
        return features

    def predict(self):
        """
        #abstract method, where actual prediction will be implemented
        #:return: list of predictions for test set
        """
        raise NotImplementedError

    def run(self):
        """
        predict salary according to implementation of model and calculate the error
        :return error of model
        """
        print('Training {}'.format(self.__class__.__name__))
        (train_result, test_result) = self.predict()
        (train_error, test_error) = self.calculate_error(train_result, test_result)
        return 1

    def calculate_error(self, train_result, test_result):
        """
        calculate mean absolute error
        :param train_result: list of predictions for train set
        :param train_result: list of predictions for test set
        """
        train_error = sklearn.metrics.mean_absolute_error(self.features.y_train, train_result)
        test_error = sklearn.metrics.mean_absolute_error(self.features.y_test, test_result)
        print("Train MSE of {}: {}".format(self.__class__.__name__, train_error))
        print("Test MSE of {}: {}".format(self.__class__.__name__, test_error))
        return train_error, test_error

import abc
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from cleaning_functions import pandas_vector_to_list

# global constants for data file names
TRAIN_NORMALIZED_LOCATION_FILE_NAME = '../data/train_normalised_location.csv'
TRAIN_DATA_CSV_FILE_NAME = '../data/Train_rev1.csv'


class BaseModel(object):
    data = None
    normalized_location_data = None

    def __init__(self, train_data_csv_file_name=TRAIN_DATA_CSV_FILE_NAME,
                 train_normalized_location_file_name=TRAIN_NORMALIZED_LOCATION_FILE_NAME,
                 train_size=0.8, test_size=None, load_location=False):
        """
        :param train_size: can be either a float or int
            - float: ratio of how much is training/test data
            - int: for total size
        :param test_size: see train_size
        :param load_location: boolean, true if we want to load and process the normalized location
        """
        if BaseModel.data is None:
            self.load_csv_data(train_data_csv_file_name)

        # split original training dataset into train and test data
        # since we most likely wont get the test salary
        self.train_data, self.test_data = train_test_split(BaseModel.data,
                                                           train_size=train_size,
                                                           test_size=test_size)
        self.training_data_size = len(self.train_data)
        self.test_data_size = len(self.test_data)

        self.train_description_feature = self.train_data[['FullDescription']]
        self.train_contract_type_feature = self.train_data[['ContractType']]
        self.train_contract_time_feature = self.train_data[['ContractTime']]
        self.train_category_feature = self.train_data[['Category']]
        self.train_company_feature = self.train_data[['Company']]
        self.train_source_name_feature = self.train_data[['SourceName']]
        self.train_location_raw_feature = self.train_data[['LocationRaw']]
        self.train_salary_normalized = self.train_data[['SalaryNormalized']]

        self.test_description_feature = self.test_data[['FullDescription']]
        self.test_salary_normalized = self.test_data[['SalaryNormalized']]

        # can be disabled to speed up load times
        if load_location:
            if BaseModel.normalized_location_data is None:
                self.load_location_data(train_normalized_location_file_name)
            train_normalized_location_data = \
                BaseModel.normalized_location_data[:self.training_data_size]

            # test_normalized_location_data = normalized_location_data[-self.test_data_size:]

            self.cleaned_town_feature = train_normalized_location_data[['town']]
            self.cleaned_region_feature = train_normalized_location_data[['region']]

    @staticmethod
    def load_location_data(train_normalized_location_file_name):
        BaseModel.normalized_location_data = pd.read_csv(train_normalized_location_file_name)

    @staticmethod
    def load_csv_data(train_data_csv_file_name):
        BaseModel.data = pd.read_csv(train_data_csv_file_name)

    @staticmethod
    def load_all_data(self):
        """
        loads all data into the static set
        useful, if this ever is going to be parallelized
        (i.e. load data first, then parallelize models)
        """
        BaseModel.load_csv_data(TRAIN_DATA_CSV_FILE_NAME)
        BaseModel.load_location_data(TRAIN_NORMALIZED_LOCATION_FILE_NAME)

    @abc.abstractmethod
    def predict_salary(self):
        """
        abstract method, where actual prediction will be implemented

        :return: list of predictions for test set
        """
        raise NotImplementedError

    def run(self):
        """
        predict salary according to implementation of model and calculate the error

        :return error of model
        """
        salary = self.predict_salary()
        error = self.calculate_error(salary)
        return error

    def calculate_error(self, predicted_salary):
        """
        calculate mean absolute error
        :param predicted_salary: list of predictions for test set
        """
        truth = self.get_truth()
        error = sklearn.metrics.mean_absolute_error(truth, predicted_salary)
        print("Mean Absolute Error: {}".format(error))
        return error

    def get_truth(self):
        """
        see return

        the purpose of this method is to let child classes overwrite the true test set
        for if it is e.g. working on a subset

        :return: list of true normalized salaries from the test set
        """
        return pandas_vector_to_list(self.test_salary_normalized)

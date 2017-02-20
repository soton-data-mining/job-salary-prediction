import math

import abc
import pandas as pd
import sklearn

from cleaning_functions import pandas_vector_to_list


class BaseModel:
    def __init__(self, train_raw_data_csv_file_name='../data/Train_rev1.csv',
                 train_normalized_location_file_name='../data/train_normalised_location.csv',
                 train_test_ratio=0.8, load_location=True):
        # TODO: make this static with single load
        self.raw_data = pd.read_csv(train_raw_data_csv_file_name)

        # split original training dataset into train and test data
        # since we most likely wont get the test salary
        self.full_dataset_size = len(self.raw_data)
        self.training_dataset_size = math.ceil(self.full_dataset_size * train_test_ratio)
        self.test_dataset_size = self.full_dataset_size - self.training_dataset_size
        self.train_raw_data = self.raw_data[:self.training_dataset_size]
        self.test_raw_data = self.raw_data[-self.test_dataset_size:]

        self.train_description_feature = self.train_raw_data[['FullDescription']]
        self.train_contract_type_feature = self.train_raw_data[['ContractType']]
        self.train_contract_time_feature = self.train_raw_data[['ContractTime']]
        self.train_category_feature = self.train_raw_data[['ContractTime']]
        self.train_company_feature = self.train_raw_data[['Company']]
        self.train_source_name_feature = self.train_raw_data[['SourceName']]
        self.train_location_raw_feature = self.train_raw_data[['LocationRaw']]
        self.train_salary_normalized = self.train_raw_data[['SalaryNormalized']]

        self.test_description_feature = self.test_raw_data[['FullDescription']]
        self.test_salary_normalized = self.test_raw_data[['SalaryNormalized']]

        # can be disabled to speed up load times
        if load_location:
            normalized_location_data = pd.read_csv(train_normalized_location_file_name)
            train_normalized_location_data = normalized_location_data[: self.training_dataset_size]
            # test_normalized_location_data = normalized_location_data[-self.test_dataset_size:]

            self.cleaned_town_feature = train_normalized_location_data[['town']]
            self.cleaned_region_feature = train_normalized_location_data[['region']]

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
        :return:
        """
        salary = self.predict_salary()
        self.calculate_error(salary)

    def calculate_error(self, predicted_salary):
        """
        calculate mean absolute error
        :param predicted_salary: list of predictions for test set
        """
        truth = pandas_vector_to_list(self.test_salary_normalized)
        error = sklearn.metrics.mean_absolute_error(truth, predicted_salary)
        print("Mean Absolute Error: {}".format(error))

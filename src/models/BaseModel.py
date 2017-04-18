import abc
import numpy as np
import os.path
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from cleaning_functions import (get_one_hot_encoded_feature,
                                get_binary_encoded_feature,
                                update_location,
                                pandas_vector_to_list,
                                remove_sub_string)
from company_feature_extraction import clean_company_name
from job_description_feature_extraction import get_one_hot_encoded_words
from job_titles_feature_extraction import get_stemmed_sentences


class BaseModel(object):
    data = None
    normalized_location_data = None

    TRAIN_NORMALIZED_LOCATION_FILE_NAME = '../data/train_normalised_location.csv'
    TRAIN_DATA_CSV_FILE_NAME = '../data/Train_rev1.csv'
    CLEANED_FILE_NAME = '../data/Binary_Preprocessed_Data.csv'

    def __init__(self, train_size=0.75, test_size=None):
        """
        :param train_size: can be either a float or int
         - float: ratio of how much is training/test data
         - int: for total size
        :param test_size: see train_size - allows you to work on a smaller data set
         i.e. 100 train, 50 test for debugging or during development
         defaults to None which equals 1-train_size
        """
        print('Initialized {}'.format(self.__class__.__name__))
        if os.path.exists(BaseModel.CLEANED_FILE_NAME):
            print('Pre-processed data exists, reading from the file')
            self.cleaned_encoded_data, self.column_names = self.load_cleaned_data()
            print('Data read complete')
        else:
            print('Pre-processed data doesn\'t exist, preprocessing data first')
            print('This operation will take a while')
            self.load_all_data()
            print('Raw data read complete')
            # Get features to clean and reshape
            self.description_feature = BaseModel.data[['FullDescription']]
            self.contract_type_feature = BaseModel.data[['ContractType']]
            self.contract_time_feature = BaseModel.data[['ContractTime']]
            self.category_feature = BaseModel.data[['Category']]
            self.company_feature = BaseModel.data[['Company']]
            self.source_name_feature = BaseModel.data[['SourceName']]
            self.location_raw_feature = BaseModel.data[['LocationRaw']]
            self.title_feature = BaseModel.data[['Title']]
            self.salary_feature = BaseModel.data[['SalaryNormalized']]
            # Read cleaned locations from separate file, cleaned with Google location
            self.cleaned_town_feature = BaseModel.normalized_location_data[['town']]
            self.cleaned_region_feature = BaseModel.normalized_location_data[['region']]

            (self.processed_data, self.feature_names) = self.preprocess_data()
            self.export_data(self.processed_data, self.feature_names, 'Binary_Preprocessed_Data')
            self.cleaned_encoded_data, self.column_names = self.load_cleaned_data()

        print('Splitting train and test')
        self.train_data, self.test_data = train_test_split(self.cleaned_encoded_data,
                                                           train_size=train_size,
                                                           test_size=test_size,
                                                           random_state=1)
        # Because 1 is good
        # Random state is there so that train and test is always the same for everyone
        self.x_train = self.train_data[:, 0:self.train_data.shape[1] - 1]
        self.y_train = self.train_data[:, self.train_data.shape[1] - 1]

        self.x_test = self.test_data[:, 0:self.test_data.shape[1] - 1]
        self.y_test = self.test_data[:, self.test_data.shape[1] - 1]
        print('Train test split complete \n')

    def preprocess_data(self):
        print('Pre-processing begins')
        # Description: consisting of 5 features existence of words below in the description:
        # 'excellent', 'graduate', 'immediate', 'junior', 'urgent'
        (onehot_encoded_desc_words, col_names_1) = \
            get_one_hot_encoded_words(self.description_feature)
        # Contract type: One hot encoded, 3 features: part time, full time, *is empty*
        (onehot_encoded_contract_type, col_names_2) = \
            get_one_hot_encoded_feature(self.contract_type_feature, 'contract_type')
        # Contract time: One hot encoded, 3 features: permanent, contract, *is empty*
        (onehot_encoded_contract_time, col_names_3) = \
            get_one_hot_encoded_feature(self.contract_time_feature, 'contract_time')
        # Company: Binary encoded
        cleaned_company = clean_company_name(self.company_feature)
        (binary_encoded_company, col_names_4) = \
            get_binary_encoded_feature(cleaned_company, 'company')
        # Source name: Binary encoded
        (binary_encoded_source, col_names_5) = \
            get_binary_encoded_feature(self.source_name_feature, 'source')
        # Town: Binary encoded
        # Region: Binary encoded
        print('Pre-processing halfway done')
        updated_town_feature = update_location(self.location_raw_feature,
                                               self.cleaned_town_feature)
        updated_region_feature = update_location(self.location_raw_feature,
                                                 self.cleaned_region_feature)
        (binary_encoded_town, col_names_6) = \
            get_binary_encoded_feature(updated_town_feature, 'town')
        (binary_encoded_region, col_names_7) = \
            get_binary_encoded_feature(updated_region_feature, 'region')
        # Job titles and modifiers: Binary encoded
        processed_job_titles, processed_job_modifiers = \
            get_stemmed_sentences(self.title_feature)
        (binary_encoded_job_titles, col_names_8) = \
            get_binary_encoded_feature(processed_job_titles, 'job_titles')
        (binary_encoded_job_modifiers, col_names_9) = \
            get_binary_encoded_feature(processed_job_modifiers, 'job_modifiers')
        # Job category: Binary encoded
        processed_category_feature = remove_sub_string('Jobs', self.category_feature)
        (binary_encoded_categories, col_names_10) = \
            get_binary_encoded_feature(processed_category_feature, 'category')

        concatenated_features = np.concatenate((
            onehot_encoded_desc_words,
            onehot_encoded_contract_type,
            onehot_encoded_contract_time,
            binary_encoded_company,
            binary_encoded_source,
            binary_encoded_town,
            binary_encoded_region,
            binary_encoded_job_titles,
            binary_encoded_job_modifiers,
            binary_encoded_categories,
            [pandas_vector_to_list(self.salary_feature)]
        ))
        concatenated_features = np.transpose(concatenated_features)

        concatenated_column_names = np.concatenate((
            col_names_1,
            col_names_2,
            col_names_3,
            col_names_4,
            col_names_5,
            col_names_6,
            col_names_7,
            col_names_8,
            col_names_9,
            col_names_10,
            ['Salary']
        ))
        print('Pre-processing finished \n')
        return (concatenated_features, concatenated_column_names)

    @staticmethod
    def export_data(list_to_write, column_names, file_name):
        print('Exporting data to ../data/' + file_name)
        print('Will read from there next time to avoid pre-processing')
        f = open('../data/' + file_name + '.csv', 'a')

        # Write column names first
        for index, item, in enumerate(column_names):
            f.write(str(item))
            if index < len(column_names) - 1:
                f.write(',')
            else:
                f.write('\n')
        # Write data
        for main_index, sub_list in enumerate(list_to_write):
            for index, item in enumerate(sub_list):
                f.write(str(item))
                if index < len(sub_list) - 1:
                    f.write(',')
            if main_index < len(list_to_write) - 1:
                f.write('\n')
        f.close()
        print('Exporting complete \n')

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
        BaseModel.data = pd.read_csv(BaseModel.TRAIN_DATA_CSV_FILE_NAME)
        BaseModel.normalized_location_data = \
            pd.read_csv(BaseModel.TRAIN_NORMALIZED_LOCATION_FILE_NAME)

    @staticmethod
    def load_cleaned_data():
        loaded_data = pd.read_csv(BaseModel.CLEANED_FILE_NAME)
        column_names = list(loaded_data.columns.values)
        loaded_data = loaded_data.as_matrix()
        loaded_data = loaded_data.astype(int)
        return (loaded_data, column_names)

    abc.abstractmethod

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
        train_error = sklearn.metrics.mean_absolute_error(self.y_train, train_result)
        test_error = sklearn.metrics.mean_absolute_error(self.y_test, test_result)
        print("Train MSE of {}: {}".format(self.__class__.__name__, train_error))
        print("Test MSE of {}: {}".format(self.__class__.__name__, test_error))
        return (train_error, test_error)

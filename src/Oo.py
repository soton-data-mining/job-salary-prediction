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



TRAIN_NORMALIZED_LOCATION_FILE_NAME = '../data/train_normalised_location.csv'
TRAIN_DATA_CSV_FILE_NAME = '../data/Train_rev1.csv'
CLEANED_FILE_NAME = '../data/Binary_Preprocessed_Data.csv'

data = pd.read_csv(TRAIN_DATA_CSV_FILE_NAME)
normalized_location_data = \
    pd.read_csv(TRAIN_NORMALIZED_LOCATION_FILE_NAME)

location_raw_feature = data[['LocationRaw']]
title_feature = data[['Title']]


processed_job_titles, processed_job_modifiers =  get_stemmed_sentences(title_feature)


def WriteBulkResults(fileName, ListOfListToWrite):
    f = open(directory+fileName, 'a')
    for list in ListOfListToWrite:
        for item in list:
            f.write(str(item))
            f.write(',')
        f.write('\n')
    f.close()

list_to_write = [ list_titles, processed_job_titles, processed_job_modifiers]


for index,item in enumerate(list_titles):
    if type(item) is str:
        list_titles[index] = item.replace(',', '-')
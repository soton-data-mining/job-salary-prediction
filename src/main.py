#!/usr/bin/env python

import pandas as pd

from job_titles_feature_extraction import get_stemmed_sentences
from job_description_feature_extraction import (get_one_hot_encoded_words,
                                                get_rake_keywords,
                                                get_top_idf_features,
                                                get_tfidf_similarity,
                                                extract_relevant_documents)
from cleaning_functions import (get_one_hot_encoded_feature,
                                get_binary_encoded_feature,
                                update_location, pandas_vector_to_list)

train_raw_data_csv_file_name = '../data/Train_rev1.csv'
train_normalized_location_file_name = '../data/train_normalised_location.csv'
test_raw_data_csv_file_name = '../data/Test_rev1.csv'

if __name__ == "__main__":
    train_raw_data = pd.read_csv(train_raw_data_csv_file_name)
    test_raw_data = pd.read_csv(test_raw_data_csv_file_name)
    train_normalized_location_data = pd.read_csv(train_normalized_location_file_name)

    train_description_feature = train_raw_data[['FullDescription']]
    train_contract_type_feature = train_raw_data[['ContractType']]
    train_contract_time_feature = train_raw_data[['ContractTime']]
    train_category_feature = train_raw_data[['ContractTime']]
    train_company_feature = train_raw_data[['Company']]
    train_source_name_feature = train_raw_data[['SourceName']]
    train_location_raw_feature = train_raw_data[['LocationRaw']]

    test_description_feature = test_raw_data[['FullDescription']]

    cleaned_town_feature = train_normalized_location_data[['town']]
    cleaned_region_feature = train_normalized_location_data[['region']]
    # Update empty town-region features with already existing features
    updated_town_feature = update_location(train_location_raw_feature, cleaned_town_feature)
    updated_region_feature = update_location(train_location_raw_feature, cleaned_region_feature)

    # Train one hot encoded features - desciption, contract-type, contract-time
    # All lists containt list of lists of features in one hot encoded format ready to append to
    # cleaned set

    # Description: consisting of 5 features existence of words below in the description:
    # 'excellent', 'graduate', 'immediate', 'junior', 'urgent'
    train_one_hot_encoded_desc_words = get_one_hot_encoded_words(train_description_feature)
    # Contract type: One hot encoded, 3 features: part time, full time, *is empty*
    train_one_hot_encoded_contract_type = get_one_hot_encoded_feature(train_contract_type_feature)
    # Contract time: One hot encoded, 3 features: permanent, contract, *is empty*
    train_one_hot_encoded_contract_time = get_one_hot_encoded_feature(train_contract_time_feature)

    # Train binary encoded features - company, source, town, region
    # Company name: Binary encoded
    train_binary_encoded_company = get_binary_encoded_feature(train_company_feature)
    # Source name: Binary encoded
    train_binary_encoded_source = get_binary_encoded_feature(train_source_name_feature)
    # Town: Binary encoded
    cleaned_binary_encoded_town = get_binary_encoded_feature(updated_town_feature)
    # Region: Binary encoded
    cleaned_binary_encoded_region = get_binary_encoded_feature(updated_region_feature)

    # get stemmed/sorted job title and job title modifiers

    processed_job_titles, processed_job_modifiers = get_stemmed_sentences(
        train_raw_data[['Title']]
    )

    train_binary_encoded_job_titles = get_binary_encoded_feature(processed_job_titles)
    train_binary_encoded_job_modifiers = get_binary_encoded_feature(processed_job_modifiers)

    keywords = get_rake_keywords(train_description_feature)
    # one_hot_encoded_features are a list of 5 lists consisting of 1s and 0s
    # If a word appears in the description, this kind of representation will
    # make it feasible for machine learning
    # We will need to append those lists on final data frame when everything is cleaned

    # get top k terms with the highest idf score (this is atm not very useful
    # but it will be used for the tf.idf similarity stuff
    top_keywords = get_top_idf_features(train_description_feature, 5)

    # calculate similarity between train an testset job descriptions
    # this is of high order complexity - test it on a subset of the data
    corpus_list = pandas_vector_to_list(train_description_feature)
    queries_list = pandas_vector_to_list(test_description_feature)
    similarity = get_tfidf_similarity(corpus_list, queries_list)
    relevant_documents = extract_relevant_documents(similarity)

import pandas as pd

from job_description_feature_extraction import get_one_hot_encoded_words, get_rake_keywords, get_top_features


raw_data_csv_file_name = '../data/Train_rev1.csv'


if __name__ == "__main__":
    raw_data = pd.read_csv(raw_data_csv_file_name)
    description_feature = raw_data[['FullDescription']]
    one_hot_encoded_features = get_one_hot_encoded_words(description_feature)
    keywords = get_rake_keywords(description_feature)
    top_keywords = get_top_features(description_feature, 5)
    # one_hot_encoded_features are a list of 5 lists consisting of 1s and 0s
    # If a word appears in the description, this kind of representation will
    # make it feasible for machine learning
    # We will need to append those lists on final data frame when everything is cleaned

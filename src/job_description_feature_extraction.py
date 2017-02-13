import string
import operator
from collections import Counter
import RAKE

JOB_DESCRIPTION_FIELD = 'FullDescription'

STOP_WORDS_PATH = "../preprocessing_data/smartstop.txt"


def get_one_hot_encoded_words(feature_to_extract):
    """
    get_feature_word_count(x) returns a list of tuples which consists of word:count
    From those lists, the words appear in the list below ( word_list ) was hand picked
    They are the word which might have an impact on the salary
    """
    word_list = ['excellent', 'graduate', 'immediate', 'junior', 'urgent']
    encoded_features = []
    for item in word_list:
        one_hot_encoded_list = []
        for data in feature_to_extract.iterrows():
            sentence = data[1][0].lower()  # Lower casing the sentence
            if item in sentence:
                one_hot_encoded_list.append(1)
            else:
                one_hot_encoded_list.append(0)
        encoded_features.append(one_hot_encoded_list)
    return encoded_features


def get_feature_word_count(column_to_count):
    count_dict = {}
    for data in column_to_count.iterrows():
        string_data = str(data)
        data_digits_removed = ''.join(
            [i for i in string_data if not i.isdigit()]
        )  # Remove digits
        puct_exclude = set(string.punctuation)
        data_punct_removed = ''.join(ch for ch in data_digits_removed if ch not in puct_exclude)
        # Remove punctuations
        data_punct_removed = data_punct_removed.lower()  # Lowercase the string
        data_as_list = data_punct_removed.split()  # Turn into list
        counts_as_dict = Counter(data_as_list)  # Turn into dictionary with counts
        for item in counts_as_dict:
            if item in count_dict:  # If item exists in the dict
                count_dict[item] += 1
            else:  # If item doesnt exists
                count_dict[item] = 1
    for item in list(count_dict):
        if count_dict[item] < len(column_to_count) / 200:  # Anything below 0.5 is insignificant
            del count_dict[item]
    count_dict = sorted(count_dict.items(), key=operator.itemgetter(0))
    # Returns a list of tuples which has word:count
    return count_dict


def get_rake_keywords(job_description):
    """
    extract keywords of documents using Rapid Automatic Keyword Extraction

    :param job_description: data frame of job_description feature from csv
    :return: list of keywords per document
    """
    job_description_list = job_description[JOB_DESCRIPTION_FIELD].values.tolist()
    keywords = []
    rake = RAKE.Rake(STOP_WORDS_PATH)
    for doc in job_description_list:
        # save top 3 results, the rest is usually garbage
        keywords.append(rake.run(doc)[:3])
    return keywords

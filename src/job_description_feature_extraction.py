import string
import operator
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

JOB_DESCIPTION_FIELD = 'FullDescription'


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


def get_top_features(job_description, k):
    """
    use TfIdf to extract top k keywords of corpus

    :param job_description: data frame
    :param k: number of
    :return: list of k top features
    """
    # based on https://stackoverflow.com/questions/25217510/
    tfidf_vectorizer, term_vector = _vectorize(job_description)
    # use _build_idf_dict to get a {term: score} dictionary which was the whole point of this exercise
    # but it makes the build fail because of unused variables so yea
    # idf_dict = _build_idf_dict(tfidf_vectorizer)
    idf_ranking = np.argsort(tfidf_vectorizer.idf_)[::-1]
    features = tfidf_vectorizer.get_feature_names()
    return [features[i] for i in idf_ranking[:k]]


def _build_idf_dict(tfidf_vectorizer):
    """
    builds a dictionarry of all terms

    :param tfidf_vectorizer
    :return: dictrionary in form of {term: score}
    """
    idf_dict = {}
    features = tfidf_vectorizer.get_feature_names()
    # TODO: not sure if that's the most efficient way to do this..
    for term, score in zip(features, tfidf_vectorizer.idf_):
        idf_dict[term] = score
    return idf_dict


def _vectorize(job_description):
    """
    vectorize job_descriptoins using tfidf

    :param job_description: data_frame
    :return: (vectorizer, term_vector)
    """
    tfidf_vectorizer = TfidfVectorizer()
    job_description_list = job_description[JOB_DESCIPTION_FIELD].values.tolist()
    term_vector = tfidf_vectorizer.fit_transform(job_description_list)
    return tfidf_vectorizer, term_vector

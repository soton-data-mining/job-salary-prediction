import string

import RAKE
import numpy as np
import operator
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    return (encoded_features, word_list)


def get_feature_word_count(column_to_count):
    # TODO: this function is never used, remove?
    # TODO: this entire function can be implemented in a single line using a count vectorizer
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


def get_tfidf_similarity(corpus, queries):
    """
    calculate cosine similarity using tf.if between corpus an queries

    :param corpus: list of text
    :param queries: list of text
    :return: matrix of ONLY corpus x query similarities
    """

    # # easier to test with smaller data set
    # # use this to overwrite the incoming corpus/queries
    # corpus = ['bob is in the house', 'susi goes to school', 'et tu brutu']
    # queries = ['where is the school', 'bob is in the house']

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus, queries)
    corpus_vector = tfidf_vectorizer.transform(corpus)
    queries_vector = tfidf_vectorizer.transform(queries)

    similarity = cosine_similarity(queries_vector, corpus_vector)

    # TODO: we can not use _vectorize here, because of the need of using the same vectorizer
    # either rewrite it or remove it entirely, if get_top_idf_features is not going to be used
    # corpus_len = len(corpus)
    # queries_len = len(queries)

    # we have to calculate the similarity for corpus + query to avoid dimension mismatch
    # but we are only interested in the similarity of queries and documents
    # remove all corpus x corpus similarities
    # similarity = similarity[-queries_len:]
    # remove all query x query similarities
    # similarity = numpy.delete(similarity, range(corpus_len, corpus_len + queries_len), 1)

    return similarity


def extract_relevant_documents(similarity, sim_threshold=0.8):
    """
    return relevant documents from similarity matrix

    :param similarity: similarity matrix from get_tfidf_similarity()
    :param sim_threshold: cutoff thershold for relevant document (0.8) by default
    :return: dict of [(document_index, similarity)] for every query from similarity matrix
    """
    relevant_documents = defaultdict(list)
    for query_index, similarity_list in enumerate(similarity):
        for document_index, similarity_score in enumerate(similarity_list):
            if similarity_score > sim_threshold:
                relevant_documents[query_index].append((document_index, similarity_score))
    return relevant_documents


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


def get_top_idf_features(job_description, k, order=-1):
    """
    use TfIdf to extract top k keywords of corpus

    :param job_description: data frame
    :param k: number of top features
    :param order: sort order of idf features (least or most significant features)
        defaults to -1 = most significant features
    :return: list of k top features
    """
    # based on https://stackoverflow.com/questions/25217510/
    tfidf_vectorizer, term_vector = _tfidf_vectorize(job_description)
    # use _build_idf_dict to get a {term: score} dictionary which was the whole point here
    # but it makes the build fail because of unused variables so yea
    # idf_dict = _build_idf_dict(tfidf_vectorizer)
    idf_ranking = np.argsort(tfidf_vectorizer.idf_)[::order]
    features = tfidf_vectorizer.get_feature_names()
    return [features[i] for i in idf_ranking[:k]]


def _build_idf_dict(tfidf_vectorizer):
    """
    builds a dictionary of all terms

    :param tfidf_vectorizer
    :return: dictionary in form of {term: score}
    """
    idf_dict = {}
    features = tfidf_vectorizer.get_feature_names()
    # TODO: not sure if that's the most efficient way to do this..
    for term, score in zip(features, tfidf_vectorizer.idf_):
        idf_dict[term] = score
    return idf_dict


def _tfidf_vectorize(job_description_list, tfidf_vectorizer=TfidfVectorizer()):
    """
    vectorize job_descriptions using tfidf

    :param job_description: list of text
    :return: (vectorizer, term_vector)
    """
    term_vector = tfidf_vectorizer.fit_transform(job_description_list)
    return tfidf_vectorizer, term_vector

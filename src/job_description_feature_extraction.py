import string

import RAKE
import numpy as np
import operator
from collections import Counter
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

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


def calc_cosine_sim(corpus_vector, queries_vector):
    """
    calculate cosine similarity given tf.idf vectors of corpus an queries

    :param corpus_vector: vectorized  ext
    :param queries_vector: vectorized text
    :return: matrix of ONLY corpus x query similarities
    """
    similarity = cosine_similarity(queries_vector, corpus_vector)
    return similarity


def cosine_knn(corpus_vector, queries_vector, k=10):
    """

    :param corpus_vector: vectorized document text
    :param queries_vector: vectorized query text
    :param k: number of neighbours
    :return: (distances, indices) of knn
    """
    # based on
    # http://scikit-learn.org/stable/modules/neighbors.html
    # http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html

    # since we want to use cosine similarity to account for document length
    # we have to use bruteforce search
    # parallelize to number of cores with n_jobs -1
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
    nbrs.fit(corpus_vector)
    distances, indices = nbrs.kneighbors(queries_vector)
    return distances, indices


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


def extract_top_k_documents(similarity, k=5):
    """
    return top k similar documents from similarity matrix

    :param similarity: similarity matrix from get_tfidf_similarity()
    :param k: top k entries so return
    :return: dict of [(document_index, similarity)] for every query from similarity matrix
    """
    indicies = []
    distances = []

    for similarity_list in similarity:
        index = [i for i in np.argsort(similarity_list)[-k:]][::-1]
        distances.append([1 - similarity_list[i] for i in index])
        indicies.append(index)

    return distances, indicies


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
    tfidf_vectorizer, term_vector, foo = tfidf_vectorize(job_description)

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


def tfidf_vectorize(documents, queries=[''],
                    tfidf_vectorizer=TfidfVectorizer(stop_words='english', lowercase=True)):
    """
    vectorize job_descriptions using tfidf

    :param documents: list of text (training_data
    :param queries: list of text (test data) - can be empty [''] (default)
        in case we just want to vectorize a single corpus
    :param tfidf_vectorizer: to overwrite with an existing/trained vectorizer
        or different parameters
    :return: (tfidf_vectorizer, document_vector, queries_vector)
    """

    # easier to test with smaller data set
    # use this to overwrite the incoming corpus/queries
    # documents = ['aaa bbb', 'ccc eee', 'aaa ddd', 'ddd ddd', 'ccc aaa']
    # queries = ['aaa bbb', 'ddd ddd']

    tfidf_vectorizer.fit(documents, queries)
    document_vector = tfidf_vectorizer.transform(documents)
    queries_vector = tfidf_vectorizer.transform(queries)
    return tfidf_vectorizer, document_vector, queries_vector

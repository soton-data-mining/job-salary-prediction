import gc
import itertools
import math
from multiprocessing.pool import Pool

import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from cleaning_functions import pandas_vector_to_list
from job_description_feature_extraction import (tfidf_vectorize,
                                                cosine_knn, calc_cosine_sim,
                                                extract_top_k_documents)
from models.BaseModel import BaseModel

DATA_QUERIES_VECTOR_NPZ = '../preprocessing_data/queries_vector.npz'
DATA_CORPUS_VECTOR_NPZ = '../preprocessing_data/corpus_vector.npz'
DATA_SIMILIARITY_MATRIX = '../preprocessing_data/similiarity_matrix'
DATA_Y_TRAIN = '../preprocessing_data/y_train'
DATA_Y_TEST = '../preprocessing_data/y_test'

FORCE_LOAD = True


class StandaloneSimilarity(BaseModel):
    def __init__(self, train_size=0.75, test_size=None, force_load_all=True):
        # overwrite constructor so we can work on cached data
        # without loading the entire data into memory
        if not os.path.exists(DATA_QUERIES_VECTOR_NPZ) or FORCE_LOAD:
            super().__init__(train_size=train_size, test_size=test_size,
                             force_load_all=force_load_all)

    def predict(self):
        if os.path.exists(DATA_QUERIES_VECTOR_NPZ) and not FORCE_LOAD:
            print('{}: loading precomputed data'.format(self.__class__.__name__))
            self.load_precomputed_data()
        else:
            self.precomputed_similarity()

        batch_size = 100
        batch_elements = math.ceil(self.queries_vector.shape[0] / batch_size)
        batch_queue = np.array_split(self.queries_vector.A, batch_elements)
        print("starting batch computation of Similarity and KNN calculation")

        # # multiple versions of calculating the prediction, some faster, some use more mem

        # prediction = self.multiprocessor_batch_calc(batch_queue)
        prediction = self.batch_calculation(batch_queue)
        # prediction = self.individual_calculation()
        # prediction = self.cosine_knn_calc()
        # prediction = self.custom_knn_calculation(prediction)

        train_avg_salary = sum(self.y_train) / len(self.y_train)
        cleaned_predictions = [x if str(x) != 'nan' else train_avg_salary for x in prediction]

        return self.y_train, cleaned_predictions

    def custom_knn_calculation(self, prediction):
        # used to calculate FULL similarity matrix and extract relevant documents from there
        # were using a much faster KNN algorithm below, achieving the similar results
        print('{}: calculating similarity martrix'.format(self.__class__.__name__))
        self.similarity_martix = calc_cosine_sim(self.corpus_vector, self.queries_vector)
        # relevant_documents = extract_relevant_documents(similarity)
        disti, indi = extract_top_k_documents(self.similarity_martix, k=5)
        prediction = self.calculate_prediction(disti, indi)
        return prediction

    def cosine_knn_calc(self):
        print('{}: calculating KNN'.format(self.__class__.__name__))
        distances, indices = cosine_knn(self.corpus_vector, self.queries_vector, k=5)
        return self.calculate_prediction(distances, indices)

    def individual_calculation(self):
        prediction = []
        pbar = tqdm(total=self.queries_vector.shape[0])
        for query in self.queries_vector:
            similarity_martix = calc_cosine_sim(self.corpus_vector, query)
            dist, ind = extract_top_k_documents(similarity_martix, k=5)
            prediction.append(self.calculate_prediction(dist, ind)[0])
            pbar.update()
        pbar.close()
        return prediction

    def batch_calculation(self, batch_queue):
        prediction = []
        pbar = tqdm(total=len(batch_queue))
        for queries in batch_queue:
            prediction.extend(self.predict_batch(queries))
            pbar.update()
        pbar.close()
        return prediction

    def multiprocessor_batch_calc(self, batch_queue):
        p = Pool(3)
        prediction = p.map(self.predict_batch, batch_queue)
        return list(itertools.chain.from_iterable(prediction))

    def predict_batch(self, queries):
        similarity_martixi = calc_cosine_sim(self.corpus_vector, queries)
        disti, indi = extract_top_k_documents(similarity_martixi, k=5)
        prediction = self.calculate_prediction(disti, indi)
        return prediction

    def calculate_prediction(self, distances, indices):
        # print('{}: calculating predictions from KNN'.format(self.__class__.__name__))
        prediction = []
        for distance, index_list in zip(distances, indices):
            # similarity equals by 1-distance
            similarity = [1 - dist for dist in distance]
            # normalize to sum up to one
            similarity_sum = sum(similarity)
            normalized_similarity = [sim / similarity_sum for sim in similarity]

            # sum up weighted average of knn dependent on similarity
            salary = 0
            for index, weight in zip(index_list, normalized_similarity):
                # TODO: maybe use some more sophisticated weighted function here?
                salary += self.y_train[index] * weight
            prediction.append(salary)
        return prediction

    def precomputed_similarity(self):
        # calculate similarity between train an testset job descriptions
        # this is of high order complexity - test it on a subset of the data
        corpus_list = pandas_vector_to_list(self.description_train_data)
        queries_list = pandas_vector_to_list(self.description_test_data)
        self.free_memory()
        print('{}: starting to vectorize description'.format(self.__class__.__name__))
        # use custom vectorizer to cut of min/max 1% of df since they carry little information
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=0.05,
                                     max_df=0.99)
        vectorizer, corpus_vector, queries_vector = tfidf_vectorize(corpus_list,
                                                                    queries_list,
                                                                    tfidf_vectorizer=vectorizer)
        print("vocabulary size: {}".format(len(vectorizer.get_feature_names())))

        self.store_precomputed_data(corpus_vector, queries_vector,
                                    self.y_train, self.y_test)
        self.load_precomputed_data()

    def free_memory(self):
        # remove references to data not used in this model to free up memory
        self.cleaned_encoded_data = 0
        self.data = 0
        self.description_train_data = 0
        self.description_test_data = 0
        self.test_data = 0
        self.train_data = 0
        self.x_test = 0
        self.x_train = 0
        self.description_feature = 0
        self.normalized_location_data = 0

        gc.collect()

    def load_precomputed_data(self):
        self.y_train = np.load(DATA_Y_TRAIN)
        self.y_test = np.load(DATA_Y_TEST)
        self.corpus_vector = self.load_sparse_csr(DATA_CORPUS_VECTOR_NPZ)
        self.queries_vector = self.load_sparse_csr(DATA_QUERIES_VECTOR_NPZ)
        # self.similarity_martix = np.load(DATA_SIMILIARITY_MATRIX)

    def store_precomputed_data(self, corpus_vector, queries_vector, y_train, y_test):
        y_train.dump(DATA_Y_TRAIN)
        y_test.dump(DATA_Y_TEST)
        self.save_sparse_csr(DATA_CORPUS_VECTOR_NPZ, corpus_vector)
        self.save_sparse_csr(DATA_QUERIES_VECTOR_NPZ, queries_vector)
        # similarity_martix.dump(DATA_SIMILIARITY_MATRIX)

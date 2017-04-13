import gc
from sklearn.feature_extraction.text import TfidfVectorizer

from cleaning_functions import pandas_vector_to_list
from job_description_feature_extraction import (tfidf_vectorize,
                                                cosine_knn)
from models.BaseModel import BaseModel


class StandaloneSimilarity(BaseModel):
    def predict(self):

        # calculate similarity between train an testset job descriptions
        # this is of high order complexity - test it on a subset of the data
        corpus_list = pandas_vector_to_list(self.description_train_data)
        queries_list = pandas_vector_to_list(self.description_test_data)

        self.free_memory()
        print('{}: starting to vectorize description'.format(self.__class__.__name__))

        # use custom vectorizer to cut of min/max 1% of df since they carry little information
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, min_df=0.01, max_df=0.99)
        vectorizer, corpus_vector, queries_vector = tfidf_vectorize(corpus_list,
                                                                    queries_list,
                                                                    tfidf_vectorizer=vectorizer)
        print("vocabulary size: {}".format(len(vectorizer.get_feature_names())))


        # # used to calculate FULL similarity matrix and extract relevant documents from there
        # # were using a much faster KNN algorithm below, achieving the similar results
        # similarity = calc_cosine_sim(corpus_vector, queries_vector)
        # relevant_documents = extract_relevant_documents(similarity)

        print('{}: calculating KNN'.format(self.__class__.__name__))
        distances, indices = cosine_knn(corpus_vector, queries_vector)

        print('{}: creating predictions'.format(self.__class__.__name__))

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

        return self.y_train, prediction

    def free_memory(self):
        # i have only have 8 gigs of memory and if we keep multiple copies of everything my laptop
        # will start swapping like crazy
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

"""
import random

from cleaning_functions import pandas_vector_to_list
from job_description_feature_extraction import get_tfidf_similarity, extract_relevant_documents

"""
from models.BaseModel import BaseModel


class StandaloneSimilarity(BaseModel):
    def predict(self):
        """
        # calculate similarity between train an testset job descriptions
        # this is of high order complexity - test it on a subset of the data
        corpus_list = pandas_vector_to_list(self.train_description_feature)
        queries_list = pandas_vector_to_list(self.test_description_feature)
        similarity = get_tfidf_similarity(corpus_list, queries_list)
        relevant_documents = extract_relevant_documents(similarity)

        # no-op functions to avoid pep8 "unused variable" error
        isinstance(relevant_documents, int)

        # return predictions, random variable for now
        return [random.randrange(20000, 30000)] * self.test_data_size
        """
        return (self.y_train, self.y_test)

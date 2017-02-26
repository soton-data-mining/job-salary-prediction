from cleaning_functions import pandas_vector_to_list
from job_description_feature_extraction import (tfidf_vectorize,
                                                document_knn)
from models.BaseModel import BaseModel


class StandaloneSimilarity(BaseModel):
    def predict_salary(self):
        # calculate similarity between train an testset job descriptions
        # this is of high order complexity - test it on a subset of the data
        corpus_list = pandas_vector_to_list(self.train_description_feature)
        queries_list = pandas_vector_to_list(self.test_description_feature)
        tfidf_vectorizer, corpus_vector, queries_vector = tfidf_vectorize(corpus_list, queries_list)

        # similarity = calc_cosine_sim(corpus_vector, queries_vector)
        # relevant_documents = extract_relevant_documents(similarity)

        distances, indices = document_knn(corpus_vector, queries_vector)

        prediction = []
        for distance, index_list in zip(distances, indices):
            # similarity equals by 1-distance
            similarity = [1 - dist for dist in distance]
            # normalize to sum up to one
            # TODO: maybe use some weighted function here?
            similarity_sum = sum(similarity)
            normalized_similarity = [sim / similarity_sum for sim in similarity]

            # having to call pandas_vector_to_list every time is fun
            train_salary_normalized = pandas_vector_to_list(self.train_salary_normalized)

            # sum up weighted average of knn dependent on similarity
            salary = 0
            for index, weight in zip(index_list, normalized_similarity):
                salary += train_salary_normalized[index] * weight
            prediction.append(salary)

        return prediction

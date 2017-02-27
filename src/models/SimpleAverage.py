from cleaning_functions import pandas_vector_to_list
from models.BaseModel import BaseModel


class SimpleAverage(BaseModel):
    """
    baseline model, just returning the average salary for each prediction
    """

    def predict_salary(self):
        # having to call pandas_vector_to_list every time is fun
        train_salary_normalized = pandas_vector_to_list(self.train_salary_normalized)
        avg_salary = sum(train_salary_normalized) / len(train_salary_normalized)
        return [avg_salary] * self.test_data_size

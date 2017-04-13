from models.BaseModel import BaseModel


class SimpleAverage(BaseModel):
    """
    baseline model, just returning the average salary for each prediction
    """
    def predict(self):
        train_avg_salary = sum(self.y_train) / len(self.y_train)
        test_avg_salary = sum(self.y_test) / len(self.y_test)
        return (([train_avg_salary] * len(self.y_train)), ([test_avg_salary] * len(self.y_test)))

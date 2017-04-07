from models.BaseModel import BaseModel
from sklearn import linear_model


class LinearRegression(BaseModel):
    def predict(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.x_train, self.y_train)
        train_result = regr.predict(self.x_train)
        test_result = regr.predict(self.x_test)
        return (train_result, test_result)

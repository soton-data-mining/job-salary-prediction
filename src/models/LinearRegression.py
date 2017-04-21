from models.BaseModel import BaseModel
from sklearn import linear_model

from models.BaseModelX import BaseModelX


class LinearRegression(BaseModel):
    def predict(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.x_train, self.y_train)
        train_result = regr.predict(self.x_train)
        test_result = regr.predict(self.x_test)
        return (train_result, test_result)


class LinearRegressionX(BaseModelX):
    def predict(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.features.x_train, self.features.y_train)
        train_result = regr.predict(self.features.x_train)
        test_result = regr.predict(self.features.x_test)
        return (train_result, test_result)

from sklearn.ensemble import RandomForestRegressor as RFR
from models.BaseModel import BaseModel


class RandomForestRegressor(BaseModel):
    def predict(self):
        
        regr_rf = RFR(max_depth= None, random_state = 9, n_estimators=100, min_samples_split = 10,
                      min_samples_leaf = 10)
        regr_rf.fit(self.x_train, self.y_train)
        train_result = regr_rf.predict(self.x_train)
        test_result = regr_rf.predict(self.x_test)
        BaseModel.export_prediction(test_result, 'RandomForestReg')
        return (train_result, test_result)

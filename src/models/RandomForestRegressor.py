from sklearn.ensemble import RandomForestRegressor as RFR
from models.BaseModel import BaseModel


class RandomForestRegressor(BaseModel):
    def predict(self):
        regr_rf = RFR(max_depth=17,
                      random_state=9,
                      n_estimators=50,
                      n_jobs=-1)
        regr_rf.fit(self.x_train, self.y_train)
        train_result = regr_rf.predict(self.x_train)
        test_result = regr_rf.predict(self.x_test)

        export_filename = 'RandomForestReg'
        if self.drop_feature_names:
            export_filename += '_without_' + '_'.join(self.drop_feature_names)

        BaseModel.export_prediction(test_result, export_filename)
        return (train_result, test_result)

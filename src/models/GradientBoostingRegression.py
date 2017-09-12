from sklearn.ensemble import GradientBoostingRegressor as GBR
from models.BaseModel import BaseModel


class GradientBoostedRegressor(BaseModel):
    def predict(self):
        print('GBTree training')
        clf = GBR(loss='quantile', alpha=0.95,
                                n_estimators=1, max_depth=2,
                                learning_rate=.1, min_samples_leaf=9,
                                min_samples_split=9)
        clf.fit(self.x_train, self.y_train)
        train_result = clf.predict(self.x_train)
        test_result = clf.predict(self.x_test)
        BaseModel.export_prediction(test_result, 'GBTree')
        return (train_result, test_result)

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as MAS
from models.BaseModel import BaseModel
import json

class RandomForestRegressor(BaseModel):
    def ml_model(self):
        print('Random Forest Starts')
        BaseModel.remove_features([])

        excluded_features = ""
        max_depth = [3, 5, 7, 10, 12, 15]
        min_sample_split = [2,5,10]
        initial_estimators = 25

        f = open('../predictions/ForestResults.csv', 'a')
        f.write('Exlcluded,depth,min_sample_split,estimators,train,test')
        f.write('\n')
        f.close()

        for depth in max_depth:
            for est_multiplier in(1,51):
                for split in min_sample_split:
                    estimator = initial_estimators * est_multiplier
                    regr_rf = RFR(random_state=9, max_depth = depth, n_estimators = estimator, \
                                  min_samples_split = split)
                    regr_rf.fit(self.x_train, self.y_train)
                    train_result = regr_rf.predict(self.x_train)
                    test_result = regr_rf.predict(self.x_test)

                    (train_error, test_error) = \
                    BaseModel.calculate_error(self, train_result, self.y_train, test_result, self.y_test)
                    print(train_error)

                    BaseModel.export_prediction(train_result,(excluded_features+"RandomForest_train"+str(train_error)+"_params:d="+
                                      str(depth)+"_estimators="+str(estimator)+"_samplesSplit="+str(split)))

                    BaseModel.export_prediction(test_result,(excluded_features+"RandomForest_test"+str(test_error)+"_params:d="+
                                    str(depth)+"_estimators="+str(estimator)+"_samplesSplit="+str(split)))


                    f = open('../predictions/ForestResults.csv', 'a')
                    f.write(excluded_features+','+str(depth)+','+str(split)+
                            ','+str(estimator)+','+str(train_error)+','+str(test_error))
                    f.write('\n')
                    f.close()
        return (self.y_train, self.y_test)

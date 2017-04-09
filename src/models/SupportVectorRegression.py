from models.BaseModel import BaseModel
import sklearn.svm as SVM

class SVR(BaseModel):
    def predict(self):
        print('Support Vector Regression training begins')
        svr_rbf = SVM.SVR(kernel='rbf', C=1e3, gamma=0.1)
        train_result = svr_rbf.fit(self.x_train, self.y_train).predict(self.x_train)
        test_result = svr_rbf.fit(self.x_train, self.y_train).predict(self.x_test)
        BaseModel.export_prediction(test_result, 'SVR_RBF_C1e3_Gamma01.csv')
        return (train_result, test_result)

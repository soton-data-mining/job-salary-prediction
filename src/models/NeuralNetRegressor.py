from models.BaseModel import BaseModel
# from keras.layers.core import Dense
# from keras.models import Sequential
from keras import optimizers as opt
# from keras import initializers
from keras import metrics
from keras.models import model_from_json
import os.path


class NeuralNetRegressor(BaseModel):
    def predict(self):

        def get_weights(model, layer_id):
            layer = model.layers[layer_id]
            weights = layer.get_weights()
            firstWeights = weights[1]
            print(firstWeights)

        def export_model(model, name):
            if not (os.path.exists("neural_net_models")):
                os.makedirs("neural_net_models")
            model_json = model.to_json()
            with open("neural_net_models/" + name + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("neural_net_models/" + name + ".h5")

        def import_model(model_name):
            json_file = open("neural_net_models/" + model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("neural_net_models/" + model_name + ".h5")
            print("Loaded " + model_name + " from disk")
            return model

        model = import_model('ut_Dense100_L1_m5s3_L2_m1s03_lr07_d1e07')
        """
        model = Sequential()
        model.add(Dense(100, input_dim=85, activation='relu',
                        kernel_initializer=initializers.RandomNormal(
                                mean=5, stddev=3, seed=None)))
        model.add(Dense(1, activation='linear',
                        kernel_initializer=initializers.RandomNormal(
                                mean=1, stddev=0.3, seed=None)))
        """
        # rms = opt.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay =1e-9)
        adadelta = opt.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        # nadam = opt.Nadam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_absolute_error', optimizer=adadelta, metrics=[metrics.mae])
        # optimizer='adam'
        model.fit(
                self.x_train, self.y_train,
                validation_data=(self.x_test, self.y_test),
                epochs=1000, batch_size=160000, verbose=1
        )

        export_model(model, 'ut_Dense100_L1_m5s3_L2_m1s03_lr07_d1e07')
        return (self.y_train, self.y_test)

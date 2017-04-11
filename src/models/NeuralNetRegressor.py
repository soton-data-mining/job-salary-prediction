from models.BaseModel import BaseModel
from keras.layers.core import Dense
from keras.models import Sequential
from keras import optimizers as opt
from keras import initializations
import numpy as np


class NeuralNetRegressor(BaseModel):
    def predict(self):

        def weight_init(shape, name=None):
            np.random.seed(1)
            return initializations.normal(shape, scale=0.02, name=name)

        model = Sequential()
        model.add(Dense(200, input_dim=18, activation='relu', init=weight_init))
        model.add(Dense(10, activation='relu', init=weight_init))
        model.add(Dense(1))
        rms = opt.RMSprop(lr=0.07, rho=0.9, epsilon=1e-08, decay=1e-8)
        # nadam = opt.Nadam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        model.compile(loss='mean_absolute_error', optimizer=rms, metrics=["accuracy"])
        # optimizer='adam'
        model.fit(self.x_train, self.y_train, validation_data=(self.x_test,
            self.y_test), nb_epoch=50, batch_size=20000, verbose=1)

        layer = model.layers[1]
        weights = layer.get_weights()
        firstWeights = weights[0]
        secWeights = weights[1]

        print(firstWeights)
        print(secWeights)
        return (self.y_train, self.y_test)

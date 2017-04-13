#!/usr/bin/env python
from models.SimpleAverage import SimpleAverage
from models.LinearRegression import LinearRegression
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity
from models.NeuralNetRegressor import NeuralNetRegressor
from models.RandomForestRegressor import RandomForestRegressor

if __name__ == "__main__":
    SA = SimpleAverage()
    SA.run()

    RF = RandomForestRegressor()
    RF.run()

    NNR = NeuralNetRegressor()
    NNR.run()

    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.run()

    SupportVectorRegressionModel = SVR()
    SupportVectorRegressionModel.run()

    StandaloneSim = StandaloneSimilarity(train_size=70000, test_size=5000)
    StandaloneSim.run()


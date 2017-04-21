#!/usr/bin/env python
from models.SimpleAverage import SimpleAverage
from models.LinearRegression import LinearRegression, LinearRegressionX
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity
from models.NeuralNetRegressor import NeuralNetRegressor
from models.RandomForestRegressor import RandomForestRegressor

if __name__ == "__main__":
    # SA = SimpleAverage()
    # SA.run()

    # RF = RandomForestRegressor()
    # RF.run()
    #
    # RFX = RandomForestRegressorX()
    # RFX.run()

    # NNR = NeuralNetRegressor()
    # NNR.run()
    #
    # LinearRegressionModel = LinearRegression()
    # LinearRegressionModel.run()


    LinearRegressionModelX = LinearRegressionX()
    LinearRegressionModelX.run()
    # example of reusing existing feature objects
    LinearRegressionModelX2 = LinearRegressionX(features=LinearRegressionModelX.features)
    LinearRegressionModelX2.run()


    #
    # SupportVectorRegressionModel = SVR()
    # SupportVectorRegressionModel.run()
    #
    # StandaloneSim = StandaloneSimilarity()
    # StandaloneSim.run()

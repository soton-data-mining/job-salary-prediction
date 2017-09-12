#!/usr/bin/env python
from models.SimpleAverage import SimpleAverage
from models.LinearRegression import LinearRegression
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity
from models.RandomForestRegressor import RandomForestRegressor
from models.GradientBoostingRegression import GradientBoostedRegressor
from models.NeuralNetRegressor import NeuralNetRegressor

if __name__ == "__main__":
    # GBTree = GradientBoostedRegressor()
    # GBTree.run()
    """
    SA = SimpleAverage()
    SA.run()
    """
    #RF = RandomForestRegressor()
    #RF.run()

    NNR = NeuralNetRegressor()
    NNR.run()
    """
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.run()

    SupportVectorRegressionModel = SVR()
    SupportVectorRegressionModel.run()

    StandaloneSim = StandaloneSimilarity()
    StandaloneSim.run()
    """

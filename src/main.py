#!/usr/bin/env python
from models.LinearRegression import LinearRegression
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity
from models.NeuralNetRegressor import NeuralNetRegressor
from models.RandomForestRegressor import RandomForestRegressor

if __name__ == "__main__":
    
    #RF = RandomForestRegressor()
    #RF.run()
    
    #NNR = NeuralNetRegressor()
    #NNR.run()
    
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.run()
    
    """
    SupportVectorRegressionModel = SVR()
    SupportVectorRegressionModel.run()



    StandaloneSim = StandaloneSimilarity()
    StandaloneSim.run()
    """
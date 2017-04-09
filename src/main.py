#!/usr/bin/env python
from models.LinearRegression import LinearRegression
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity

if __name__ == "__main__":

    SupportVectorRegressionModel = SVR()
    SupportVectorRegressionModel.run()
    
    #LinearRegressionModel = LinearRegression()
    #LinearRegressionModel.run()

    #StandaloneSim = StandaloneSimilarity()
    #StandaloneSim.run()

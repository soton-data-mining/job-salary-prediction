#!/usr/bin/env python
from models.LinearRegression import LinearRegression
from models.StandaloneSimilarity import StandaloneSimilarity

if __name__ == "__main__":
    
    LinearRegressionModel = LinearRegression()
    LinearRegressionModel.run()

    StandaloneSim = StandaloneSimilarity()
    StandaloneSim.run()

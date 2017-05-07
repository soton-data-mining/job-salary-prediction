#!/usr/bin/env python
from models.SimpleAverage import SimpleAverage
from models.LinearRegression import LinearRegression
from models.SupportVectorRegression import SVR
from models.StandaloneSimilarity import StandaloneSimilarity
from models.NeuralNetRegressor import NeuralNetRegressor
from models.RandomForestRegressor import RandomForestRegressor

from results_viewer.ResultsGrapher import ResultsGrapher

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

    StandaloneSim = StandaloneSimilarity()
    StandaloneSim.run()

    # models_to_graph = [RF]
    models_to_graph = [SA, RF, NNR, LinearRegressionModel,
                       SupportVectorRegressionModel, StandaloneSim]

    rg = ResultsGrapher(models_to_graph)
    rg.plot_bar_chart_error_rates_all()

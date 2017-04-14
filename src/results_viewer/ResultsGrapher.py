import numpy as np
import matplotlib.pyplot as plt


class ResultsGrapher:

    def __init__(self, models):
        self.models = models

    def plot_bar_chart_error_rates_all(self):
        names = [m.__class__.__name__ for m in self.models]
        y_pos = np.arange(len(names))
        performance = [m.mae_test_error for m in self.models]
        plt.bar(y_pos, performance, align='center', alpha=0.5)
        plt.xticks(y_pos, names)
        plt.ylabel('Error')
        plt.title('Model Used')
        plt.show()

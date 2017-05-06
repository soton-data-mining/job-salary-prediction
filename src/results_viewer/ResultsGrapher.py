import numpy as np
import matplotlib.pyplot as plt


class ResultsGrapher:
    def __init__(self, models):
        self.models = models

    def plot_bar_chart_error_rates_all(self):
        n_groups = len(self.models)
        means_train_errors = [m.mae_train_error for m in self.models]
        means_test_errors = [m.mae_test_error for m in self.models]
        names = [m.__class__.__name__ for m in self.models]
        plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        plt.bar(index, means_train_errors, bar_width, alpha=opacity,
                color='b',
                label='Train error')

        plt.bar(index + bar_width, means_test_errors, bar_width,
                alpha=opacity,
                color='g',
                label='Test error')

        plt.xlabel('Models Used')
        plt.ylabel('MAE Error')
        plt.title('MAE erros per model')
        plt.xticks(index + bar_width, names)
        plt.legend()

        plt.tight_layout()
        plt.show()

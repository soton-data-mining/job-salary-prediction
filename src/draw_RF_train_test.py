from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

rf_max_depth = [3,5,7,9,11,13,15,17,19,21]
rf_avg_train_MAE = [12767,12020, 11361, 10597, 10043, 9121, 8124, 7241, 6240, 4351 ]
rf_avg_test_MAE = [12849, 12015, 11462, 10566, 10148, 9401, 8742, 8421, 8612, 8901 ]
average_line = [11324]*10

"""
plt.gca().set_color_cycle(['blue', 'black', 'red'])
plt.plot(rf_max_depth, rf_avg_train_MAE,
         rf_max_depth, rf_avg_test_MAE,
         rf_max_depth, average_line, linewidth=3.0 )
#plt.title('Random Forest Tree Depth over MAE', fontsize=15)
plt.ylabel('Mean Absolute Error', fontsize=15)
plt.xlabel('Tree depth', fontsize=15)
red_patch = mpatches.Patch(color='red', label='Average')
blue_patch = mpatches.Patch(color='blue', label='Train')
black_patch = mpatches.Patch(color='black', label='Test')
plt.legend(handles = [red_patch, blue_patch, black_patch ], loc=3)
plt.savefig('depth.png')
"""



tree_estimators = [30,60,90,120,150,200,300,400]
rf_est_train_mae = [7539, 7502, 7469, 7492, 7367, 7294, 7265, 7262]
rf_est_test_mae =  [8898, 8865, 8853, 8850, 8875, 8857, 8854, 8855]

plt.gca().set_color_cycle(['blue', 'black'])
plt.plot(tree_estimators, rf_est_train_mae,
         tree_estimators, rf_est_test_mae, linewidth=3.0 )
#plt.title('Random Forest Estimators over MAE(depth=15)', fontsize=15)
plt.ylabel('Mean Absolute Error', fontsize=15)
plt.xlabel('Count of Estimators(Trees)', fontsize=15)
blue_patch = mpatches.Patch(color='blue', label='Train')
black_patch = mpatches.Patch(color='black', label='Test')
plt.legend(handles = [blue_patch, black_patch ], loc=5)
plt.savefig('foo.png')

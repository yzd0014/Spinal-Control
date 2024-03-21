import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datafile = './datalog/baesline-processed.pkl'
data = pd.read_pickle(datafile)
data_baseline = data[data['Model'] == 'baseline']
data_sac = data[data['Model'] == 'sac']
value_baseline = data_baseline.Value.to_numpy()
value_sac = data_sac.Value.to_numpy()
all_data = [value_baseline, value_sac]
labels = ['baseline', 'sac']
plt.boxplot(all_data, labels=labels)
plt.show()


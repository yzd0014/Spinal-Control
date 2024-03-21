import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datafile = './datalog/baesline-processed.pkl'
data = pd.read_pickle(datafile)
data = data[data['Model'] == 'baseline']
value = data.Value.to_numpy()
all_data = [value, value]
labels = ['baseline', 'sac']
plt.boxplot(all_data, labels=labels)
plt.show()


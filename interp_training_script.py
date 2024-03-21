import pandas as pd
from plotnine import *
import scipy.stats as stats
import numpy as np
import pickle

df_file = './datalog/training_log.pkl'
savefile = 'baesline-processed.pkl'
value_to_compare = 'rollout/ep_len_mean'
value_string = 'Episode Length'
fs = 20


df = pd.read_pickle(df_file)
v_df = df[df['Tag'] == value_to_compare]

full_steps = list(range(0,v_df['Step'].max() + 1))
all_data = []
for model in v_df['Model'].unique():
  for modelid in v_df[v_df['Model'] == model]['ModelID'].unique():
    sub_df = v_df[(v_df['Model'] == model) & (v_df['ModelID'] == modelid)]
    sub_df = sub_df.set_index('Step').reindex(full_steps).reset_index()
    sub_df['Value'] = sub_df['Value'].interpolate(method='linear')
    original_max_step = (v_df[(v_df['Model'] == model)
                              & (v_df['ModelID'] == modelid)]['Step'].max())
    sub_df.loc[sub_df['Step'] > original_max_step, 'Value'] = np.nan
    sub_df['Model'] = model
    sub_df['ModelID'] = modelid
    all_data.append(sub_df)
interp_df = pd.concat(all_data, ignore_index=True)
output = interp_df[interp_df['Step'] == 60000]
pickle.dump(output, open('./datalog/' + savefile, 'wb'))



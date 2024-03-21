import pickle
import plot_training_data as pt

datafile = 'baeline-processed.pkl'

data = pickle.load(open(datafile,'rb'))

interp_df = data['interp_df']
v_df = data['v_df']
means = data['means']
maxes = data['maxes']
pt.plot_data(interp_df,v_df,means,maxes,'Episode Length')

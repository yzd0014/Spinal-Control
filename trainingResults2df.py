import os
import sys
from tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import pickle
# Path to your TensorBoard log files
log_path = ['./logs']
all_data = []

for l in log_path:
  for f in os.listdir(l):
    subdir_path = os.path.join(l, f)
    if os.path.isdir(subdir_path):
      for sf in os.listdir(subdir_path):
        subsubdir_path = os.path.join(subdir_path, sf)
        if os.path.isdir(subsubdir_path):
          events = \
              [os.path.join(subsubdir_path, f) \
              for f in os.listdir(subsubdir_path) if 'tfevents' in f]
        for event in events:
          event_acc = EventAccumulator(event)
          event_acc.Reload()
          if 'scalars' in event_acc.Tags():
            scalar_tags = event_acc.Tags()['scalars']
            for tag in scalar_tags:
              data = event_acc.Scalars(tag)
              df = pd.DataFrame(data=[(s.step, s.value) for s in data],
                                columns=['Step', 'Value'])

              model_str = os.path.basename(subdir_path)
              model = model_str.split('-')

              df['Tag'] = tag
              df['ModelID'] = model_str
              df['Model'] = '-'.join(model[1:])
              all_data.append(df)  # Add the DataFrame to our list

df = pd.concat(all_data, ignore_index=True)
file_name = "training_log.pkl"
pickle.dump(df, open("./datalog/" + file_name, 'wb'))

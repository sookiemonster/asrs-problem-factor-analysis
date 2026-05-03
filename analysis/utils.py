from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import MultiLabelBinarizer
import os

# SETTING UP DATASETS

def split_set(df:pd.DataFrame, target:str="Assessments_Primary_Problem"):
  assert target in df.columns

  X = df[[col for col in df.columns if col != target]]
  y = df[target]

  return X, y

def make_balanced(X:pd.DataFrame, y:pd.Series, downsample_to_freq:int):
  """
  Accepts an X,y and selects at most `downsample_to_freq` instances for each label in y
  or just uses all instances for label if count(instances w/ label) < downsample_to_freq.
  """
  c = y.name
  df = pd.merge(X, y, left_index=True, right_index=True)
  li = []

  for cat in y.unique():
    matched = df[df[c] == cat]
    li.append(matched.head(downsample_to_freq))

  res = pd.concat(li)
  res = res.sample(frac=1, random_state=42) # shuffle
  res.to_csv("balanced.csv")
  return split_set(res, c)
  
# EVALUATION

def visualize_eval(y_true, pred, label:str) -> None:
  # Metrics
  print(classification_report(y_true, pred))

  # Confusion Matrix
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12))

  ConfusionMatrixDisplay.from_predictions(
      y_true,
      pred,
      xticks_rotation='vertical',
      ax=ax1
  )
  ax1.set_title("Raw Counts")

  ConfusionMatrixDisplay.from_predictions(
      y_true,
      pred,
      xticks_rotation='vertical',
      normalize='true',
      values_format='.1f',
      ax=ax2
  )
  ax2.set_title("Normalized")

  plt.tight_layout()

  now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  plt.savefig(f"{label}_{now}.png")
  plt.show()


# Function which I moved from the NB classifier to onehot the data 
def read_and_onehot_events(csv_name : str) -> pd.DataFrame:
  path = os.path.dirname(__file__) + "/" + csv_name  
  csv_data_df = pd.read_csv(path)
  csv_data_df.rename(columns={"Unnamed: 0":"ACN"}, inplace=True)

  # Aggregate the events into one list to be used by the onehot encoder
  aggregate_events_list = csv_data_df[csv_data_df.columns[2:]].agg(list, axis=1)

  # Drop NaN entries (there's probably a cleaner way to do this)
  for i in range(len(aggregate_events_list)): 
      for j in range(len(aggregate_events_list[i])):
          if not isinstance(aggregate_events_list[i][j], str):
              aggregate_events_list[i] = aggregate_events_list[i][:j]
              break
  
  # One-hot the list of event lists
  mlb = MultiLabelBinarizer()
  onehotted = mlb.fit_transform(aggregate_events_list)
  onehot_events_df = pd.DataFrame(onehotted, columns=mlb.classes_)

  # Combine into one dataframe with the ACN and primary problem
  onehot_events_df = pd.concat([csv_data_df[csv_data_df.columns[:2]], onehot_events_df], axis=1)
  onehot_events_df.set_index("ACN", inplace=True)

  return onehot_events_df
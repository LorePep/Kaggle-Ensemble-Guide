import pandas as pd
import sys
from scipy.stats import pearsonr, kendalltau, spearmanr

first_file = sys.argv[1]
second_file = sys.argv[2]

TRAIN_CSV_PATH = "/Users/lorenzo/kaggle/whales/dataset/train.csv"



def corr(first_file, second_file, labels_to_idx):
  first_df = pd.read_csv(first_file)
  second_df = pd.read_csv(second_file)
  # assuming first column is `prediction_id` and second column is `prediction`
  prediction = first_df.columns[1]
  
  first_prediction = []
  second_prediction = []

  for _, row in first_df.iterrows():
    pred = row[prediction]
    labels_list = pred.split(" ")
    if len(labels_list) < 5:
      print(labels_list)
    first_prediction.extend([labels_to_idx[l] for l in labels_list])

  for _, row in second_df.iterrows():
    pred = row[prediction]
    labels_list = pred.split(" ")
    if len(labels_list) < 5:
      print(labels_list)
    second_prediction.extend([labels_to_idx[l] for l in labels_list])
  
  # correlation
  print("Finding correlation between: {} and {}".format(first_file,second_file))
  print("Column to be measured: {}".format(prediction))
  print("Pearson's correlation score: {}".format(pearsonr(first_prediction, second_prediction)))
  print("Kendall's correlation score: {}".format(kendalltau(first_prediction, second_prediction)))
  print("Spearman's correlation score: {}".format(spearmanr(first_prediction, second_prediction)))


train_df = pd.read_csv(TRAIN_CSV_PATH)
labels_to_idx = {label: i for i, label in enumerate(sorted(train_df["Id"].unique()))}

corr(first_file, second_file, labels_to_idx)

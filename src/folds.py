import config
import pandas as pd
from sklearn import model_selection

# if __name__ == "__main__":
#     df = pd.read_csv(config.TRAINING_FILE)
#     df["kfold"] = -1

#     df = df.sample(frac=1).reset_index(drop=True)

#     kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

#     for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.sentiment.values)):
#         print(len(train_idx), len(val_idx))
#         df.loc[val_idx, 'kfold'] = fold
#     print(df.head())
#     df.to_csv("../input/train_folds.csv", index=False)
#     print(df['kfold'].value_counts())
#     print(len(df))
#     print(df.head())

df = pd.read_csv(config.TRAINING_FILE)
print(df['kfold'].value_counts())
print(len(df[(df['sentiment'] == 'positive') & (df['kfold'] == 0)]))
print(len(df[(df['sentiment'] == 'positive') & (df['kfold'] == 3)]))
print(len(df[(df['sentiment'] == 'positive') & (df['kfold'] == 4)]))
print(len(df))
df = pd.read_csv("../input/train.csv")
print(len(df))
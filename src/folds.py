import config
import pandas as pd
from sklearn.model_selection import KFold
FOLDS = 5
SEED = 0
kf = KFold(FOLDS, random_state=SEED)
df = pd.read_csv(config.TRAINING_FILE)
df = df.sample(frac=1).reset_index(drop=True)

for f, (train_index, val_index) in enumerate(kf.split(df, df)): #4 fold fortrain, 1 fold for validation every time i take only validation fold
    print(f, type(f))
    df.loc[val_index, 'kfold'] = f


df.to_csv('../input/train_folds.csv', index = False)   

print(df['kfold'].value_counts())
print(len(df))

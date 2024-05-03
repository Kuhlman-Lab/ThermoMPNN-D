import pandas as pd

from sys import argv

df = pd.read_csv(argv[1])

df = df.loc[df.ddG_true < -0.5]
print(df.shape)
from scipy.stats import spearmanr

print(spearmanr(df.ddG_true, df.ddG_pred)[0])
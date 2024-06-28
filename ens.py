import pandas as pd 

from sys import argv

files = argv[1:]

df_list = [pd.read_csv(f) for f in files]

pred_list = []
for df in df_list:
    try:
        pred_list.append(df.ddG_additive)
    except AttributeError:
        pred_list.append(df.ddG_pred)
    
avg = pd.concat(pred_list, axis=1).mean(axis=1)
print(avg.shape)

# n_muts = df.mut_type.str.count(';') + 1
# df = df.loc[n_muts > 2]
# avg = avg.loc[n_muts > 2]

# df = df.loc[n_muts < 3]
# avg = avg.loc[n_muts < 3]

# avg = avg.loc[df.ddG_true < -0.5]
# df = df.loc[df.ddG_true < -0.5]


print(df.shape)

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import root_mean_squared_error as rmse

print(pearsonr(avg.values, df.ddG_true.values)[0])
print(spearmanr(avg.values, df.ddG_true.values)[0])
print(rmse(avg.values, df.ddG_true.values))
# %%

import pandas as pd

df = pd.read_excel("/home/yuval/Documents/Weizmann/python-cest/lp30_normalized.xlsx")


for col in df.columns[1:]:
    df[col] /= df[col].max()
df

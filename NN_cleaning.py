import pandas as pd

pd.options.mode.chained_assignment = None

df = pd.read_csv("LBW_Dataset.csv")

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
fence_low  = Q1-1.5*IQR
fence_high = Q3+1.5*IQR
median = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].median()
df["BP"] = df["BP"].mask(df["BP"] >fence_high["BP"], median["BP"])
df["BP"] = df["BP"].mask(df["BP"] < fence_low["BP"], median["BP"])
    
df['Education'].fillna(df['Education'].mode()[0], inplace=True)
df['Residence'].fillna(df['Residence'].median(), inplace=True)
df['BP'].fillna(df['BP'].mean(), inplace=True)
df['HB'].fillna(df['HB'].median(), inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Weight'].fillna(df['Weight'].median(), inplace=True)
df['Delivery phase'].fillna(df['Delivery phase'].mode()[0], inplace=True)

df.to_csv("LBW_Dataset_Preprocessed.csv")    
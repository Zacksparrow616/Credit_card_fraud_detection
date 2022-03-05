import pandas as pd 
df=pd.read_csv('creditcard.csv')
#print(df.head())
print(df['V1'].max())
print(df['V1'].min())
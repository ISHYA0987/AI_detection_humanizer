import pandas as pd 

df=pd.read_csv("data/raw/ai_human_content_detection_dataset.csv")
df1=pd.read_excel("data/raw/Dataset.xlsx")\
df2=pd.read_csv("data/raw/train.csv")
df3=pd.read_csv("data/raw/shuffled_Human.csv")



print("--Dataset 1 stats--")
print(df.info())
print(df.columns)
print("Missing values")
print(df.isnull().sum())
print("Duplicated values")
print(df.duplicated().sum())


print("--Dataset 2 stats--")
print(df1.info())
print(df1.columns)
print("Missing values")
print(df1.isnull().sum())
print("Duplicated values")
print(df1.duplicated().sum())


print("--Dataset 3 stats--")
print(df2.info())
print(df2.columns)
print("Missing values")
print(df2.isnull().sum())
print("Duplicated values")
print(df2.duplicated().sum())



print("--Dataset 4 stats--")
print(df3.info())
print(df3.columns)
print("Missing values")
print(df3.isnull().sum())
print("Duplicated values")
print(df3.duplicated().sum())

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("winequality-red.csv",sep=';') 

qq=df['quality']
col=list(df.columns) 


for i in col:
    a=df[i]
    print("Mean ",a.mean())
    print("Median ",a.median())
    print("Mode ",a.mode())
    print("Maximum ",a.max())
    print("Minimum ",a.min())
    print("Standard Deviation ",a.std())
    print("\n")
    
for i in col:
    a=df[i]
    plt.scatter(qq,a)
    plt.show()

for i in col:
    a=df[i]
    print(i," ",ff.corr(a),"\n")

for i in col:
    a=df[i]
    plt.hist(a)
    plt.show()
 
df.groupby('quality').hist("pH")


b=[]
for i in col:
    b.append(df[i])
    plt.boxplot(df[i])
    plt.show()
    

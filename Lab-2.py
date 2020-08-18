import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/sharpshooter/Desktop/DS3-Assignments/landslide_data2_miss.csv")
count1=[]
count2=[]
count3=[]
count4=[]
count5=[]
count6=[]
count7=[]
count8=[]

df1 = df.isnull()

df1.dropna(axis=0 , how="any",thresh=5)

for index, row in df1.iterrows():
    count=0
    for j in row:
        if j==True:
            count+=1
    if count==1:
        count1.append(count)
    if count==2:
        count2.append(count)
    if count==3:
        count3.append(count)
    if count==4:
        count4.append(count)
    if count==5:
        count5.append(count)
    if count==6:
        count6.append(count)
    if count==7:
        count7.append(count)
    if count==8:
        count8.append(count)
    
A=[]
for i in range(8):
    A.append(i+1)
B=[len(count1),len(count2),len(count3),len(count4),len(count5),len(count6),len(count7),len(count8)]
plt.bar(A,B)
plt.show()
    
total=0
for i in range(3,8):
    total+=B[i]
    

print("\nNo. of tuples more than 50% missing ",total)
        

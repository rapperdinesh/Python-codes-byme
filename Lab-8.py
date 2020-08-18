import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn import metrics
from sklearn import cluster
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures as pf
df=pd.read_csv("/home/rahul/Downloads/winequality-red.csv")
x_train,x_test=train_test_split(df,test_size=0.3,random_state=42,shuffle=True)
y_train=x_train["quality"]
y_test=x_test["quality"]
x_train=x_train[["pH"]]
x_test=x_test[["pH"]]
model=lr()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(metrics.mean_squared_error(y_test,y_pred)**0.5)
model1=lr()
model=pf(degree=2)
x_train=model.fit_transform(x_train)
x_test=model.fit_transform(x_test)
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)
print(metrics.mean_squared_error(y_test,y_pred)**0.5)
plt.scatter(y_test,y_pred)




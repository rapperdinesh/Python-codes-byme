###################################################################################

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix as confusion_mat
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from prettytable import PrettyTable
from sklearn.decomposition import PCA

"""
Try to use the functionalities of the libraries imported.
For example, rather than converting Pandas dataframe into
a list and then perform calculations, use methods of Pandas
library.
"""
###################################################################################


###################################################################################

# Import or Load the data
def load_dataset(path_to_file):
    
    df=pd.read_csv(path_to_file)
    return df
    
    """
    Load the dataset using this function and then return the
    dataframe. The function parameters can be as per the code
    and problem. Return the loaded data for preprocessing steps.
    """
###################################################################################


###################################################################################

# Data Preprocessing (Use only the required functions for the assignment)
"""
- Check for outliers.
- Check for missing values.
- Encoding categorical data
- Standardization/Normalization
- Dimensionality Reduction (PCA)
- Shuffle
- Train/Test Split
"""

def outliers_detection(function_parameters):
    ...
    ...
    ...

    pass
def missing_values(function_parameters):
    ...
    ...
    ...
    pass
    
def encoding(function_parameters):
    """
    Encode the categorical data in your dataset using One-Hot
    encoding. Very important if your dependent variable is
    categorical.
    """
    pass
    
def normalization(dataframe):
    
    min_max_scaler = MinMaxScaler(feature_range =(0, 1))
        
    A=list(dataframe.columns)

    class_frame= dataframe[A[len(A)-1]]
    
    dataframe.drop(A[len(A)-1], axis=1, inplace=True)
    A=A.remove('class')
  
    x_scaled = min_max_scaler.fit_transform(dataframe)
    
    dataframe = pd.DataFrame(x_scaled,columns=A)
    
    dff = pd.merge(dataframe,class_frame,left_index=True,right_index=True)

    return dff
    
    
def standard(dataframe):
    
    B=list(dataframe.columns)
    
    class_frame = dataframe[B[len(B)-1]]
    dataframe.drop(B[len(B)-1], axis=1, inplace=True)
    
    B=B.remove('class')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    
    dataframe = pd.DataFrame(scaled_data,columns=B)
    
    dff = pd.merge(dataframe,class_frame,left_index=True,right_index=True)
    
    return dff
    
    
def shuffle(dataframe):
    
    dataframe = shuffle(dataframe)
    

def train_test_split(dataframe):
    
    class_new=dataframe['class']
    dataframe=dataframe.drop(columns=["class"])

    X_train, X_test, y_train, y_test = ttsplit(dataframe, class_new, test_size=0.3, random_state=42)

    Data=[X_train, X_test, y_train, y_test]
    return Data
    

###################################################################################


###################################################################################

def percentage_accuracy(Dataset,y_predicted):
    

    accu=accuracy_score(Dataset, y_predicted)
    return accu


def confusion_matrix(Dataset,y_predicted):
    
    results = confusion_mat(Dataset, y_predicted)
    print('Confusion Matrix :')
    print(results)
    


# Perform classification

A=[]
i=0

def classification(Data):
    
    x = PrettyTable()
    x.field_names = ["K-Value", "Accuracy"]

    Accuracy_scores=[]
    
    X_trained=Data[0]
    X_tested=Data[1]
    y_trained=Data[2]
    y_tested=Data[3]
    
    
    C=[]
    for i in range(1,23):
        if i%2!=0:
            C.append(i)

    for i in C:
#        print("\n-----------------------------------------------------------")
#        print("\nFor ",i," nearest neighbors\n")
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_trained,y_trained)
        y_pred = model.predict(X_tested)

#        print("Accuracy Score: ",round(percentage_accuracy(y_tested,y_pred),2))
        Accuracy_scores.append(round(percentage_accuracy(y_tested,y_pred)*100,2))
        print("________________________")
        print("\nK value",i)
        print("\n")

        confusion_matrix(y_tested,y_pred)

        x.add_row([i,round(percentage_accuracy(y_tested,y_pred)*100,2)])

        

    A.append(Accuracy_scores)
    print(x)
    
    plt.plot(C,Accuracy_scores)
    plt.show()
    print("\nKNN Maximum Accuracy :",round(max(Accuracy_scores),2),"%")



def classifyNaiveBayes(Data):
    
    X_trained=Data[0]
    X_tested=Data[1]
    y_trained=Data[2]
    y_tested=Data[3]
    
    model = GaussianNB()
    model.fit(X_trained,y_trained)
    
    expected = y_tested
    predicted = model.predict(X_tested)

    print("Naive Bayes Accuracy Score: ",round(percentage_accuracy(expected, predicted)*100,2),"%")

#    print("\n")
#    print(confusion_matrix(expected,predicted))


def classifyGMM(Data,i):
    
    X_trained=Data[0]
    X_tested=Data[1]
    y_trained=Data[2]
    y_tested=Data[3]
    
    model = GaussianMixture(n_components=i)
    model.fit(X_trained,y_trained)
    
    expected = y_tested
    predicted = model.predict(X_tested)

    print("\nGNN Accuracy Score for ",i," : ",round(percentage_accuracy(expected, predicted)*100,2),"%")
    confusion_matrix(expected,predicted)



def operationsOnReducedData(dataframe):
    classification(dataframe)
    A=[2,4,8,16]
    for i in A:
        classifyGMM(dataframe,i)                         
                              
  

def dimensionality_reduction(dataframe,i):
    
    print("------------------------------------------------------------------")

    print("\n\nFor Dimensions: ",i)
    pca = PCA(n_components=i)
    newdata = pca.fit_transform(dataframe)
    return newdata


###################################################################################





def main():
    
    dataframe = pd.read_csv('pima-indians-diabetes.csv')
    
    df_norm=dataframe.copy()
    
    norm_data = normalization(df_norm)
#    cols=dataframe.columns
#    norm_data.rename(columns=cols)
        
##########################################################################################
    #  Normal Data
    
    print("-------------------------Normal DATA Analysis-------------------------------\n\n")    
        
    NORMAL_SPLITS= train_test_split(norm_data)
    
    print("---------------------------K-Nearest Neighbour Method ------------------------")    
    classification(NORMAL_SPLITS)
    
    
    print("---------------------------GMM Classifier Method------------------------------")  
    
    A=[2,4,8,16]
    for i in A:
        classifyGMM(NORMAL_SPLITS,i)
    
    
    print("\n---------------------------AFTER PCA ANALYSIS---------------------------------\n")
        
  
    for i in range(8):
        if i>0:
            pca_part=norm_data.copy()
            classdf = pca_part["class"]
#            print(pca_part)
            pca_new = pca_part.drop(columns=['class'])
#            print(pca_new)
           
            newdataframe = dimensionality_reduction(pca_new,i)
#            print(newdataframe)
            new=pd.DataFrame(newdataframe)
#            print(new)
           
            dff = pd.merge(new,classdf,left_index=True,right_index=True)
            dff_trainy = train_test_split(dff)   
            
            operationsOnReducedData(dff_trainy)

    
           
            
    
if __name__=="__main__":
	main()


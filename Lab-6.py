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
    A=A.remove('Class')
  
    x_scaled = min_max_scaler.fit_transform(dataframe)
    
    dataframe = pd.DataFrame(x_scaled,columns=A)
    
    dff = pd.merge(dataframe,class_frame,left_index=True,right_index=True)

    return dff
    
    
def standard(dataframe):
    
    B=list(dataframe.columns)
    
    class_frame = dataframe[B[len(B)-1]]
    dataframe.drop(B[len(B)-1], axis=1, inplace=True)
    
    B=B.remove('Class')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataframe)
    
    dataframe = pd.DataFrame(scaled_data,columns=B)
    
    dff = pd.merge(dataframe,class_frame,left_index=True,right_index=True)
    
    return dff
    
    
def shuffle(dataframe):
    
    dataframe = shuffle(dataframe)
    
    """
    Now your data is preprocessed. Shuffle to 'randomize' the data
    for next step of machine learning. Pass the respective parameters
    needed by the function and shuffle the data. Then return the
    shuffled data for next step of splitting it into training and test
    data.
    """

def train_test_split(dataframe):
    
    class_new=dataframe['Class']
    dataframe=dataframe.drop(columns=["Class"])

    X_train, X_test, y_train, y_test = ttsplit(dataframe, class_new, test_size=0.3, random_state=84)

    Data=[X_train, X_test, y_train, y_test]
    return Data
    

    """
    Now your data is preprocessed and shuffle. It's time to divide it
    into training and test data.
    
    Example:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    ...                                 test_size=0.3, random_state=42)

            X: independent features
            Y: dependent features
            test_size: fraction of data to be splitted into test data
            random_state: a seed for random generator to produce same
                            "random" results each time the code is run.

    Now your data is ready for classification.
    """

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
    for i in range(1,22):
        if i%2!=0:
            C.append(i)

    for i in C:
        print("\n-----------------------------------------------------------")
        print("\nFor ",i," nearest neighbors\n")
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_trained,y_trained)
        y_pred = model.predict(X_tested)

        print("Accuracy Score: ",round(percentage_accuracy(y_tested,y_pred),2))
        Accuracy_scores.append(round(percentage_accuracy(y_tested,y_pred),2))
        confusion_matrix(y_tested,y_pred)
        x.add_row([i,round(percentage_accuracy(y_tested,y_pred),4)])

        

    A.append(Accuracy_scores)
    print(x)
    print("\nMaximum Accuracy :",max(Accuracy_scores))

    plt.plot(C,Accuracy_scores)
    plt.show()
    
    
    
    """
    Pass the respective function parameters and perform classification.
    """

def classifyNaiveBayes(Data):
    
    X_trained=Data[0]
    X_tested=Data[1]
    y_trained=Data[2]
    y_tested=Data[3]
    
    model = GaussianNB()
    model.fit(X_trained,y_trained)
    
    expected = y_tested
    predicted = model.predict(X_tested)

    print("Accuracy Score: ",percentage_accuracy(expected, predicted))
    print("\n")
    print(confusion_matrix(expected,predicted))



def operationsOnReducedData(dataframe):
    classification(dataframe)
    classifyNaiveBayes(dataframe)                          
                              
  

def dimensionality_reduction(dataframe,i):

    print("For Dimensions: ",i)
    pca = PCA(n_components=i)
    newdata = pca.fit_transform(dataframe)
    return newdata


    """
    Pass the respective function parameters needed by the function
    and perform dimentionality reduction. Retain the useful and
    significant principal components. Dimensionality reduction
    using PCA comes at a cost of interpretibility. The features in
    original data (age, height, income, etc.) can be intrepreted
    physically but not principal components. So decide accordingly.
    Then return the dimension reduced data.
    """
                        
###################################################################################


###################################################################################

# Calculate model evaluation scores like
"""
- Accuracy
- Confusion Matrix
"""
pass

###################################################################################





def main():
    
    dataframe = pd.read_csv('DiabeticRetinipathy.csv')
    
    df_norm=dataframe.copy()
    norm_data = normalization(df_norm)
    
        
##########################################################################################
    #  Normal Data
    
    print("-------------------------Normal DATA Analysis-------------------------------\n\n")    
        
    NORMAL_SPLITS= train_test_split(norm_data)
    
    print("---------------------------K-Nearest Neighbour Method ------------------------")    
    classification(NORMAL_SPLITS)
    
    print("\n\n------------------------Naive Bayes Method----------------------------------\n\n")
    classifyNaiveBayes(NORMAL_SPLITS)
        
    for i in range(len(norm_data.columns)):
        if i>0:
            pca_part=norm_data.copy()
            classdf=pca_part["Class"]
            pca_part.drop(columns=["Class"])
            
            
            newdataframe = dimensionality_reduction(pca_part,i)
            
            df_newnorm = pd.DataFrame(newdataframe)
            dff = pd.merge(df_newnorm,classdf,left_index=True,right_index=True)
            
            
            operationsOnReducedData(dff)
            
    
if __name__=="__main__":
	main()


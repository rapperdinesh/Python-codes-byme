import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(path_to_file):
    df=pd.read_csv(path_to_file)
    return df

    """ Returns Pandas dataframe for given csv file

	
		Parameters
		----------
		path_to_file: string
			Given csv file
		
		Returns
		-------
		pandas.Dataframe
	"""
    pass
 
 

def show_box_plot(attribute_name,dataframe):
    
    df=dataframe[attribute_name]
    plt.boxplot(df)  
    plt.xlabel(attribute_name)
    plt.show()
    
    q1=df.quantile(0.25)
    
    q3=df.quantile(0.75)
    print("quartiles ",q1,q3)
    iqr=q3-q1
    count=0
    for i in df:
        if i>=(q3+(1.5*iqr)) or i<=(q1-(1.5*iqr)):
            count+=1
    
    print("No of ouliers in ",attribute_name," are ",count)
    
    
    """ Displays boxplot for atrribute

		Parameters
		----------
		attribute_name: string
			Attribute selected
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		None
	"""
    pass


def replace_outliers(dataframe):
    
    A=["temperature","humidity","rain" ]
    
    for i in A:
        kf=dataframe[i]
        med=np.median(kf)
        q1=kf.quantile(0.25)
        q3=kf.quantile(0.75)
        iqr=q3-q1
        
        kf.mask(kf >(q3+(1.5*iqr)), med,inplace=True)
        kf.mask(kf <(q1-(1.5*iqr)), med,inplace=True)
    return dataframe
    
    """ Replaces the outliers in the given dataframe
	
		Parameters
		----------
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		pandas.Dataframe
    """
    pass

def range(dataframe,attribute_name):
    
    kf=dataframe[attribute_name]
    df_range=[kf.max(),kf.min()]
    return df_range
    
    
    """ Gives Range of Selected Attribute
	
		Parameters
		----------
		attribute_name: string
			Attribute selected
		dataframe: pandas.Dataframe
			DataFrame for the given dataset
		Returns
		-------
		pair(float,float)
	"""
    pass

def min_max_normalization(df):
    
     
     df=(df-df.min())/(df.max()-df.min())

     return df
                 
             
     
     """ Returns normalized pandas dataframe
    	
    		Parameters
    		----------
    		dataframe: pandas.Dataframe
    			Dataframe for the given dataset
    		range: pair(float,float) 
    			Normalize between range
    		Returns
    		-------
    		pandas.Dataframe
    	"""
     pass
 
def min_max_normalization_set(df,maxi,mini):
    
     
     df=(((df-df.min())/(df.max()-df.min()))*(maxi-mini))+mini

     return df
                 
             
     
     """ Returns normalized pandas dataframe
    	
    		Parameters
    		----------
    		dataframe: pandas.Dataframe
    			Dataframe for the given dataset
    		range: pair(float,float) 
    			Normalize between range
    		Returns
    		-------
    		pandas.Dataframe
    	"""
     pass
 
def standardize(df):
    
     df=(df-df.mean())/df.std()

     return df
 
     """ Returns standardized pandas dataframe
	
		Parameters
		----------
		dataframe: pandas.Dataframe
			Dataframe for the given dataset
		Returns
		-------
		pandas.Dataframe
	"""
     pass

def main():
    
    """ Main Function
		Parameters
		----------
		
		Returns
		-------
		None
    """
    path_to_file="landslide_data2_original.csv"
    
    dataframe=read_data(path_to_file)
    
    dataframe=dataframe[["temperature","humidity","rain"]]



    A=["temperature","humidity","rain" ]
    

    
    for i in A:
        show_box_plot(i,dataframe)

    dataframe=replace_outliers(dataframe)
    for i in A:
        kf=dataframe[i]
        print(i)
        print("max",kf.max()," min",kf.min())
        
    print("\n\n\nThe new BOXPLOTS after replaceing are: \n\n ----------|-----------")

    for i in A:
        show_box_plot(i,dataframe)
    
    print(min_max_normalization(dataframe))
  #  print(min_max_normalization_set(dataframef,20,0))
    print(dataframe.mean(axis = 0, skipna = True))
    print(dataframe.std(axis = 0, skipna = True))
   
    
    

    da=standardize(dataframe)
    print(da)
    print(da.mean(axis = 0, skipna = True))
    print(da.std(axis = 0, skipna = True))
    
 
if __name__=="__main__":
	main()
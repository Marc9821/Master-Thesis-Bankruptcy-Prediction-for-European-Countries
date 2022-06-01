from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


def impute_macro(filepath, filename):
    # read in the dataset
    df = pd.read_csv(filepath+filename, na_values='n.a.', index_col=False)
    df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
    
    # print number of missing values
    print(f'Missing values: {df.isna().sum().sum()}')
    
    # print information on dataset
    print(df.info())
    
    # impute dataset with KNNImputer
    imputer = KNNImputer(n_neighbors=3, missing_values=np.nan, weights='distance')
    imputer = imputer.fit(df.iloc[:,1:])
    df.iloc[:,1:] = imputer.transform(df.iloc[:,1:])
    print(df.info)
    
    # print new number of missing values
    print(f'Missing values: {df.isna().sum().sum()}')
    
    # save dataset
    df.to_csv(filepath+'imputed_macro.csv')

if __name__ == '__main__':
    impute_macro('PATH_TO_FILE', 'FILENAME')
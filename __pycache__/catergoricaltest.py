# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:06:08 2022

@author: robins83
"""
import pandas as pd
from imput import ABDGImput
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


import os 
import glob

def getOriginals(path):
    originalPath={}
    for l in path:
        tableName=l.split('\\')[-1].split('.')[0]
        path=l
        originalPath[tableName]=path 
    return originalPath
    
listOfPath=glob.glob('/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/Complete_Data/*.csv')
originalPath=getOriginals(listOfPath)
directory = '/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/Incomplete_Data/Data_1'
imputeDirectory='/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/Imputed_Data/Data_1/'
AE_excel=pd.read_excel(r'/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/NRMS/NRMS.xlsx',header=None)
AEArray=AE_excel.to_numpy()    

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        o=f.split('\\')[1].split('_')[0]
        missingDFName=f.split('\\')[1].split('.')[0]
        originalDataPath=originalPath[o]
        excelPath=f
        
        
        loc=np.where(AEArray==missingDFName)
        row=loc[0][0]
        col=loc[1][0]


        imputeDir=imputeDirectory+missingDFName+'.csv'


        dataset = pd.read_excel (excelPath,header=None)
        original = pd.read_excel(originalDataPath,header=None)

        X = dataset.iloc[:,:-1]
        imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        X_imputed = pd.DataFrame(imputer.fit_transform(X))
        X_imputed.columns = X.columns

        X_original = original.iloc[:,:-1]
        y = dataset.iloc[:,-1]

        comparison_values = X.isnull()
        rows,cols=np.where(comparison_values==True)

        enc = OrdinalEncoder()
        X_enc = enc.fit_transform(X_imputed)
        X_enc_new = pd.DataFrame(X_enc)
        
        for item in zip(rows,cols):
          X_enc_new.iloc[item[0], item[1]] = np.nan

        X_enc = X_enc_new.to_numpy()
        
        if y.dtypes == 'object' :
            enc_y = OrdinalEncoder()
            y_enc = enc_y.fit_transform(np.array(y).reshape(-1,1))
            y_enc_new = pd.DataFrame(y_enc)
            y_enc_new1 = y_enc_new.squeeze()

        abdg = ABDGImput(categorical_features='auto', n_iter=4, alpha=0.6, L=0.5,
                        update_step=10, random_state=None)
        if y.dtypes == 'object' :
            abdg.fit(X_enc_new, y_enc_new1)
            X_imp, y_imp = abdg.predict(X_enc_new, y_enc_new1)
        else:
            abdg.fit(X_enc_new, y)
            X_imp, y_imp = abdg.predict(X_enc_new, y)
            
        X_imp.round(0)
        
        if y.dtypes == 'object' :
            reverse_data_y = enc_y.inverse_transform(np.array(y_imp).reshape(-1,1))
            y_imp_reversed = pd.DataFrame(reverse_data_y)

        reverse_data_x = enc.inverse_transform(X_imp)
        X_imp_reversed = pd.DataFrame(reverse_data_x)

        comp=(X_original.to_numpy()==X_imp_reversed.to_numpy())
        comp = pd.DataFrame(comp).replace({True:1,False:0})
        sumOfV=comp.values.sum()
        total=comp.count().sum()
        sumOfV,total
        AE=round(sumOfV/total,4)
        
        AEArray[row][col+2]=AE
        X.to_excel(imputeDir, index = False,header=None) 
        
AEDF=pd.DataFrame(AEArray)    
AEDF.to_csv(r'/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/NRMS/updatedNRMS.csv',index=False)        



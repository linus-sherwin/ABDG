# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 10:52:40 2022

@author: robins83
"""

import pandas as pd
from imput import ABDGImput
import numpy as np
import math
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

def nrms(e,o):
    ndf=e-o
    ndf=ndf**2
    ddf=o**2
    num=math.sqrt(ndf.sum())
    den=math.sqrt(ddf.sum())
    nrms=num/den
    return nrms

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
        
        totaldata = dataset.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        X_original = original.iloc[:,:-1]

        numeric_data = totaldata.select_dtypes(include=[np.number])
        categorical_data = totaldata.select_dtypes(exclude=[np.number])

        imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
        X_imputed = pd.DataFrame(imputer.fit_transform(categorical_data))
        X_imputed.columns = categorical_data.columns

        comparison_values = categorical_data.isnull()
        rows,cols=np.where(comparison_values==True)

        enc = OrdinalEncoder()
        X_enc = enc.fit_transform(X_imputed)
        X_enc_new = pd.DataFrame(X_enc)

        for item in zip(rows,cols):
            X_enc_new.iloc[item[0], item[1]] = np.nan
            
        X_enc_new.columns = categorical_data.columns

        X = pd.concat([numeric_data, X_enc_new],axis = 1)
        
        if y.dtypes == 'object' :
            enc_y = OrdinalEncoder()
            y_enc = enc_y.fit_transform(np.array(y).reshape(-1,1))
            y_enc_new = pd.DataFrame(y_enc)
            y_enc_new1 = y_enc_new.squeeze()

        abdg = ABDGImput(categorical_features='auto', n_iter=4, alpha=0.6, L=0.5,
                        update_step=10, random_state=None)
        if y.dtypes == 'object' :
            abdg.fit(X, y)
            X_imp, y_imp = abdg.predict(X, y_enc_new1)
        else:
            abdg.fit(X, y)
            X_imp, y_imp = abdg.predict(X, y)

        X_cat = X_imp[X_enc_new.columns]
        X_original_cat = X_original[X_enc_new.columns]
        X_cat.round(0)
        reverse_data_x = enc.inverse_transform(X_cat)
        X_cat_imp_reversed = pd.DataFrame(reverse_data_x)
        X_cat_imp_reversed.columns = categorical_data.columns

        comp=(X_original_cat.to_numpy()==X_cat_imp_reversed.to_numpy())
        comp = pd.DataFrame(comp).replace({True:1,False:0})
        sumOfV=comp.values.sum()
        total=comp.count().sum()
        sumOfV,total
        AE=round(sumOfV/total,4)


        X_num = X_imp[numeric_data.columns]
        X_original_num = X_original[numeric_data.columns]
        n = nrms(X_num.to_numpy(),X_original_num.to_numpy())

        X_imp_temp = pd.concat([X_num, X_cat_imp_reversed],axis = 1)
        X_imp_new = X_imp_temp.sort_index(axis=1)
        
        AEArray[row][col+1]=round(n,4)
        AEArray[row][col+2]=AE
        X_imp_new.to_excel(imputeDir, index = False,header=None) 
        
AEDF=pd.DataFrame(AEArray)    
AEDF.to_csv(r'/Users/linussherwin/Downloads/ABDG_Project/My_Proj_Files/Code/NRMS/updatedNRMS.csv',index=False)           
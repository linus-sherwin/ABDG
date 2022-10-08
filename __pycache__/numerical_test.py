# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:11:25 2022

@author: jimst
"""

import pandas as pd
from imput import ABDGImput
import math
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

import os 
import glob

def getOriginals(path):
    originalPath={}
    for l in path:
        tableName=l.split('\\')[-1].split('.')[0]
        #print(tableName)
        path=l
        originalPath[tableName]=path 
        #print(originalPath[tableName])
    return originalPath

def nrms(e,o):
    ndf=e-o
    ndf=ndf**2
    ddf=o**2
    num=math.sqrt(ndf.sum())
    den=math.sqrt(ddf.sum())
    nrms=num/den
    return nrms


listOfPath=glob.glob('/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Complete Data/*.csv')
originalPath=getOriginals(listOfPath)
directory = '/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Incomplete Data/Data_1'
imputeDirectory='/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Imputed_Data/Data 1/'
nrms_excel=pd.read_excel('/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/NRMS/NRMS.xlsx',header=None)
nrmsArray=nrms_excel.to_numpy()


for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        o=f.split('\\')[1].split('_AE')[0]
        print(o)
        missingDFName=f.split('\\')[1].split('.')[0]
        print(missingDFName)
        originalDataPath=originalPath[o]
        excelPath=f
        
        
        loc=np.where(nrmsArray==missingDFName)
        row=loc[0][0]
        col=loc[1][0]


        imputeDir=imputeDirectory+missingDFName+'.csv'
        dataset = pd.read_csv (excelPath,header=None)
        original = pd.read_csv(originalDataPath,header=None)

        X = dataset.iloc[:,:-1]
        X_original = original.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        
        if y.dtypes == 'object' :
            enc_y = OrdinalEncoder()
            y_enc = enc_y.fit_transform(np.array(y).reshape(-1,1))
            y_enc_new = pd.DataFrame(y_enc)
            y_enc_new1 = y_enc_new.squeeze()
            
        abdg = ABDGImput(categorical_features='auto', n_iter=4, alpha=0.6, L=0.5,
                        update_step=100, random_state=None)
        
        if y.dtypes == 'object' :
            abdg.fit(X, y_enc_new1)
            X_imp, y_imp = abdg.predict(X, y_enc_new1)
        else:
            abdg.fit(X, y)
            X_imp, y_imp = abdg.predict(X, y)

        comparison_values = X.isnull()
        rows,cols=np.where(comparison_values==True)
        for item in zip(rows,cols):
            X.iloc[item[0], item[1]] = X_imp.iloc[item[0], item[1]]

        n = nrms(X.to_numpy(),X_original.to_numpy())
        
        nrmsArray[row][col+1]=round(n,4)
        X_imp.to_excel(imputeDir, index = False,header=None) 
        
nrmsDF=pd.DataFrame(nrmsArray)    
nrmsDF.to_csv('/Users/linussherwin/Downloads/ABDG Project/Final Sample/Final/NRMS_AE/updatedNrms.csv',index=False)

 







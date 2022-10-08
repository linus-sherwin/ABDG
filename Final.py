# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:11:25 2022

@author: robins83

"""

import pandas as pd
from imput import ABDGImput
import math
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import time
import PySimpleGUI as sg

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

#Code snippet for UI
sg.theme('TealMono')
layout = [  [sg.Text('Missing Data imputation using Attribute based decision graphs', size=(100, 2),font=("Arial", 18),text_color='black')],
            [sg.Text('Enter the name of the data that needs to be imputed',font=("Helvetica", 15)), sg.InputText(font=("Helvetica", 15))],
            [sg.Button('Save',font=("Helvetica", 15)), sg.Button('Impute',font=("Helvetica", 15))] ]
window = sg.Window('Attribute Based Decision Graph', layout,size=(700, 200))
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Impute': 
        break
    Data =  values[0]
window.close()


listOfPath=glob.glob('/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Complete Data/*.csv')
originalPath=getOriginals(listOfPath)
directory = '/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Incomplete Data/' +Data
imputeDirectory='/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Imputed Data/'+Data+'/'
nrms_excel=pd.read_excel('/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/NRMS/NRMS.xlsx',header=None)
nrmsArray=nrms_excel.to_numpy()


for filename in os.listdir(directory):
    start = time.time()
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        pa = os.path.dirname(f)
        o = os.path.basename(pa)
        #print(o)
        mf = os.path.basename(f)
        missingDFName=mf.split("\\")[0].split(".")[0]
        #print(missingDFName)
        originalDataPath=originalPath['/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/Complete Data/'+o]
        excelPath=f
        
        
        loc=np.where(nrmsArray==missingDFName)
        row=loc[0][0]
        col=loc[1][0]


        imputeDir=imputeDirectory+missingDFName+'.csv'
        
        dataset = pd.read_csv(excelPath,header=None)
        original = pd.read_csv(originalDataPath,header=0)

        X = dataset.iloc[:,:-1]
        X_original = original.iloc[:,:-1]
        y = dataset.iloc[:,-1]
        
        if y.dtypes == 'object' :
            enc_y = OrdinalEncoder()
            y_enc = enc_y.fit_transform(np.array(y).reshape(-1,1))
            y_enc_new = pd.DataFrame(y_enc)
            y_enc_new1 = y_enc_new.squeeze()
            
        abdg = ABDGImput(categorical_features='auto', n_iter=3, alpha=0.6, L=0.5,
                        update_step=100, random_state=None)
        keyparams = 'categorical_features=auto, n_iter=3, alpha=0.6, L=0.5,update_step=100, random_state=None'
        
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
        end = time.time()
        runtime = end-start
        
        nrmsArray[row][col+1]=round(n,4)
        nrmsArray[row][col+2]=round(runtime,4)
        nrmsArray[row][col+3]=keyparams
        X_imp.to_csv(imputeDir, index = False,header=None) 
        
nrmsDF=pd.DataFrame(nrmsArray)    
nrmsDF.to_csv('/Users/linussherwin/Downloads/ABDG Project/My Proj Files/Code/NRMS/updatedNRMS.csv',index=False)

 







import numpy as np
import os
import nibabel as ni
import pandas as pd
import matplotlib.pyplot as plt
#-*- coding : utf-8-*-
# coding:unicode_escape

dataname = pd.read_csv("/data/subject_list.csv")
datalabel = pd.read_csv("/data/TADPOLE_D1_D2_bl.csv")
biomarkerlabel = pd.read_csv("/data/volume_ADNI.csv")



output = []
datalabel = pd.DataFrame(datalabel)
biomarkerlabel = pd.DataFrame(biomarkerlabel)
X_data = datalabel['PTID']
subject_data = [] 
x_biomarker = np.array(biomarkerlabel)


datalist = []
labellist = []


ICV = biomarkerlabel[' Intracranial volume (mL)']




volume_all = []

for i in range(len(biomarkerlabel)):
    print(ICV[i])
    volume_sub = []
    for j in range(2,77):
      volume_norm = biomarkerlabel.iloc[[i],[j]]/ICV[i]
      volume_sub.append(np.array(volume_norm))
    volume_all.append(np.array(volume_sub))
    
    
volume_all = np.array(volume_all)
volume_all = volume_all[:,:,0,0]

#save corrected volume for brain regions as V-biomarkers
datalist = pd.DataFrame(volume_all)    
datalist.to_csv("data/   .csv")
    
    
    
    

       
       


# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:32:37 2020

@author: an
"""

# --------------data preprocess--------------------------
import pandas as pd
data = pd.read_csv("train_200.csv")


data = data.drop(columns=['hour', 'id', 'site_id',  
              'app_id',  'device_ip', 'device_model',
              'C14', 'C17', 'C19' , 'C20', 'C21', 'site_domain'],axis=1)
y = data["click"]
x = data.drop(columns=["click"], axis=1)
col_name = x.columns.values
for i in col_name:
    if x[i].dtype != 'object':
        x[i] = x[i].astype('object')
# one-hot encoding        
x = pd.get_dummies(x)  

data = pd.concat([y,x], axis=1)
print("1. Converting the data......")
data.to_csv("train_200_converted.csv",index = False, header=True)

# print feature field and its corresponding feature number
col_name_converted = x.columns.values
field = dict()
for cname_old in col_name:
    if cname_old not in field:
        field[cname_old] = 0
    for cname_new in col_name_converted:
        flag = True
        for i in range(len(cname_old)):
            if cname_old[i] != cname_new[i]:
                flag = False
                break
        if flag == True and cname_new[len(cname_old)] == "_":
            field[cname_old] += 1

print("2. field and number")
for i,j in field.items():
    print(i+": "+str(j))

print("3. name of field")
print(field.keys())
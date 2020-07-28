# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 17:26:22 2020

@author: aad
"""

#unstructred data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

file_list = glob.glob(os.path.join(os.getcwd(), "Chat Transcripts for Project_", "*.text"))

text=[]

for file_path in file_list:
    with open(file_path, encoding="utf8") as f_input:
        lines = f_input.readlines()
        
        for line in lines:
            if line.startswith( '(' ):
                aa= line.strip()
                text.append(aa)

un_df= pd.DataFrame(text)
un_df.to_csv('unstructred_data.csv')
un_df.to_csv('unstructred_data.txt', index='false')

text1=[]
text2=[]                
for file_path in file_list:
    with open(file_path, encoding="utf8") as f_input:
        lines = f_input.readlines()
        
        for line in lines:
            if line.startswith( '(' ): 
                if "Mounica Patel" in line or "Ananya" in line:
                    ba= line.strip()
                    text1.append(ba)
                else:
                    ab=ba= line.strip()
                    text2.append(ab)

# mounica chats only
un_df1= pd.DataFrame(text1)
un_df1.to_csv('Mounica_chats.csv' , index='false')
un_df1.to_csv('Mounica_chats.txt', index='false')

# mVisitors chats only
un_df2= pd.DataFrame(text2)
un_df2.to_csv('visitor_chats.csv' , index='false')
un_df2.to_csv('visitor_chats.txt', index='false')

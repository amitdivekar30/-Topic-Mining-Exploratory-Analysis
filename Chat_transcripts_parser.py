# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:28:06 2020

@author: aad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

file_list = glob.glob(os.path.join(os.getcwd(), "Chat Transcripts for Project_", "*.text"))

timestamp=[]
Visitor_ID= []
Unread=[]
Visitor_Name=[]
Visitor_Email=[]
Visitor_Notes=[]
IP=[]
Country_Code=[]
Country_Name=[]
Region=[]
City=[]
User_Agent=[]
Platform=[]
Browser= []

for file_path in file_list:
    with open(file_path, encoding="utf8") as f_input:
        lines = f_input.readlines()
        
                
        for line in lines:
            if re.match("Timestamp", line):
                a= line.strip()
                timestamp.append(a[11:])
            if re.match("Visitor ID", line):
                b= line.strip()
                Visitor_ID.append(b[11:])
            if re.match("Unread", line):
                c= line.strip()
                Unread.append(c[7:])
            if re.match("Visitor Name", line):
                d= line.strip()
                Visitor_Name.append(d[14:])
            if re.match("Visitor Email", line):
                e= line.strip()
                Visitor_Email.append(e[15:])
            if re.match("Visitor Notes", line):
                f= line.strip()
                Visitor_Notes.append(f[15:])
            if re.match("IP", line):
                g= line.strip()
                IP.append(g[4:])
            if re.match("Country Code", line):
                h= line.strip()
                Country_Code.append(h[14:])
            if re.match("Country Name", line):
                k= line.strip()
                Country_Name.append(k[14:])
            if re.match("Region", line):
                l= line.strip()
                Region.append(l[8:])
            if re.match("City", line):
                m= line.strip()
                City.append(m[6:])
            if re.match("User Agent", line):
                n= line.strip()
                User_Agent.append(n[12:])
            if re.match("Platform", line):
                p= line.strip()
                Platform.append(p[10:])
            if re.match("Browser", line):
                q= line.strip()
                Browser.append(q[9:]) 

df1=[timestamp, Unread, Visitor_ID, Visitor_Name, Visitor_Email, Visitor_Notes, IP, Country_Code, Country_Name,
     Region,City,User_Agent,Platform,Browser]
df= pd.DataFrame(df1, index= ['timestamp', 'Unread', 'Visitor_ID', 'Visitor_Name', 'Visitor_Email', 'Visitor_Notes', 'IP', 'Country_Code', 'Country_Name',
     'Region','City' ,'User_Agent','Platform', 'Browser'])
df= df.T      
        
df.to_csv('structred_data.csv')

# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:09:32 2017

@author: ddo003
"""


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import scale
import os
import sys
import time

def progress_bar(total, increment, i):
    sys.stdout.write("\r")
    sys.stdout.write("["+"#"*(i//increment)+"-"*((total - i)//increment)+"]")

def get_path(name = "choose a directory"):
    """Quick function to get a gui to choose path"""
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.askdirectory(title = name)
#    os.chdir(path)
    return path

def get_filename(name = "choose a file"):
    root = tk.Tk()
    root.withdraw()
    filename = tk.filedialog.askopenfilename(title = name)
    return filename

def calculate_displacement(x,y):
    """calculates euclidian diplacement from starting coordinates when given columns with x and y coordinates"""
    dist = [None]
    
    start = x.dropna().index[0]
    
    x1 = x[start]
    y1 = y[start]
    for ii in range(1,len(x)):
        
        x2 = x[ii-1]
        y2 = y[ii-1]
        distance = np.sqrt(((x2-x1)**2)+(y2-y1)**2)
        dist.append(distance)
    return pd.Series(dist)

def obtain_M(X, Y, window):
    """returns normalized embedding matrix M for columns X and Y with the specified window size.
    This matrix can be passed to the get_H function to compute the complexity value"""
    Mx = np.array(X[:window]) #initialize first row of Mx
    My = np.array(Y[:window]) #initialize first row of My
    for ii in range(1, len(X)-window): #skip first entry since we already have that in M
        Mx = np.vstack([Mx, X[ii:ii+window]]) #add new vector to Mx
        My = np.vstack([My, Y[ii:ii+window]]) #add new vector to My
    cols = Mx.shape[1] #get number of columns from array object
    
    for ii in range(cols): #normalize per column:
        Mx[:,ii] = Mx[:,ii] - np.mean(Mx[:,ii])
        My[:,ii] = My[:,ii] - np.mean(My[:,ii])
    M = np.dstack([Mx,My]) #stack the arrays Mx and My   
    return M #return M


def get_H(M):
    """Performs singular value decomposition on M, and uses the diagonal matrix S
    to calculate complexity value H as the entropy in the distribution of components of S
    I advise you to read Herbert-Read (2017) on escape path complexity"""
    U,S,V = np.linalg.svd(M) # do singular value decomposition
    hats_array = [s/np.sum(s) for s in S] #make hats array
    local_H = [-np.sum(s*np.log2(s)) for s in hats_array]
    H = -np.sum([s*np.log2(s) for s in hats_array]) #calculate H
    return local_H,H



def distance_to_line(x,y,x_s,y_s,x_f,y_f):
    """returns the distance of point x,y to line (x_s,y_s)(x_f,y_f)"""
    d = np.abs((y_f-y_s)*x - (x_f -x_s)*y + x_f*y_s -y_f*x_s)/np.sqrt((y_f-y_s)**2+(x_f-x_s)**2)
    return d

path = get_path(name = "Choose data directory")
os.chdir(path)

print("Retrieving files, initializing parameters...")
total = 8
increment  = 1
i = 0

#progress_bar(total, increment, i)


files = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files if name.endswith(".txt") and "ages" not in name.lower() and "error" not in name.lower()]
filenames = [f.split("\\")[-1] for f in files]

i+=1
#progress_bar(total, increment, i)

#get attributes like age and dechorionation
chor = pd.Series([f.split("_")[3].upper() for f in filenames])
age = pd.Series([f.split("_")[4].upper() for f in filenames])

i+=1
#progress_bar(total, increment, i)

light = pd.Series([f.split("_")[6] for f in filenames])
drug = pd.Series([f.split("_")[1] for f in filenames])

i+=1
#progress_bar(total, increment, i)

state = pd.Series([f.split("_")[2] for f in filenames])
fps = pd.Series([30  if "10fps" not in f.lower() else 10 for f in files])
temp = pd.Series([18 if "18deg" in f.lower() else 14 for f in files])

i+=1
#progress_bar(total, increment, i)

dfs = pd.Series([pd.read_pickle(f) for f in files])


i+=1
#progress_bar(total, increment, i)
crowdsize = pd.Series([f.split("_")[5] for f in filenames])

i+=1
#progress_bar(total, increment, i)

alldata = pd.DataFrame()
alldata["chor"] = chor
alldata["age"] = age
alldata["filename"] = filenames
alldata["filepath"] = files
alldata["state"] = state
alldata["dfs"] = dfs
alldata["drug"] = drug
alldata["light"] = light
alldata["fps"] = fps
alldata["temp"]= temp
#crowdsize?
alldata["crowdsize"] = crowdsize


i+=1
#progress_bar(total, increment, i)

#modafinil = []
#for file in files:
#    if "modafinil" in file.lower():
#        if "20mgl" in file.lower():
#            modafinil.append("20mgL")
#        elif "2mgl" in file.lower():
#            modafinil.append("2mgL")
#        elif "dmso" in file.lower():
#            modafinil.append("dmso")
#        elif "200mgl" in file.lower():
#            modafinil.append("200mgL")
#        else:
#            modafinil.append("none")
#    else:
#        modafinil.append("none")
#
#alldata["modafinil"] = modafinil
#
#
#i+=1
#progress_bar(total, increment, i)

print("\n")

total = len(alldata.dfs)
increment = total//20


#Define lazy:
lazy = []
tto = []
tdo = []
ac = []
complexity = []
turn_variance = []
turn5_variance = []
speed_variance = []
mean_speed = []
cumdist = []
cumdistnorm = []
max_speed = []
good_frames= []

print("\n Iterating over all dataframes... \n")
i = 0
for df in alldata.dfs:
    #progress_bar(total, increment, i)
    
    fps = alldata.fps.iloc[i]
    
    i+=1
    
    df["Displacement"] = calculate_displacement(df.X_zero, df.Y_zero)
    
    #laziness
    if df.Displacement.max() > 900:
        lazy.append(False)
    else:
        lazy.append(True)
    
    #thigmo by time
    tto_df = len(df.loc[(df.X_zero.notnull()) & (df.Thigmo == True)])/len(df.loc[df.X_zero.notnull()])
    tto.append(tto_df)   
    
    #thigmo by distance
    if len(df[df.Thigmo == True]) == 0:
        tdo.append(0)
    elif len(df[df.Thigmo == True]) == len(df):
        tdo.append(1)
    else:
        total_dist = df.Dist_corr.dropna().cumsum().values.tolist()[-1]
        dist_out = df.Dist_corr.loc[df.Thigmo == True].dropna().sum()
        tdo.append((dist_out/total_dist))
    
    #resting coefficient    
    len_slow = len(df[df.Filtered_Speed < 100])
    len_true = len(df[df.X_zero.notnull() == True])
    r = 1-(len_slow/len_true)
    ac.append(r)
    
    #complexity
    complexity.append(df.lH.mean())

    #variance in turns
    turn_variance.append(df.Turn.var())
    turn5_variance.append(df.Turn5.var())
    #variance in speed
    speed_variance.append(df.Filtered_Speed.var())
    
    #total traveled distance:
    cumdist.append(df.Dist_corr.sum())
    
    #normalized distance
    cumdistnorm.append((df.Dist_corr.sum() / len(df[df.Dist_corr.notnull()])))
    
    #mean and max speed
    mean_speed.append(df.Filtered_Speed.mean())
    max_speed.append(df.Filtered_Speed.max())
    
    #good frames 
    gf = len(df[df.Filtered_Speed.notnull() == True])
    good_frames.append(gf)

    
alldata["lazy"] = lazy
alldata["tto"] = tto
alldata["tdo"] = tdo
alldata["ac"] = ac
alldata["complexity"] = complexity
alldata["tv"] = turn_variance
alldata["tv5"] = turn5_variance
alldata["sv"] = speed_variance
alldata["mean_speed"] = mean_speed
alldata["cumdist"] = cumdist
alldata["cumdistnorm"] = cumdistnorm
alldata["max_speed"] = max_speed
alldata["good_frames"] = good_frames





print(alldata.columns)
try:
    import pickle
    name = time.strftime("%Y%m%d")+"_alldata.pickle"
    with open(name, "wb") as f:
        print("Pickling data to 'alldata.pickle'")
        pickle.dump(alldata, f)
        print("...Done")
except:
    print("Could not pickle alldata dataframe")
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:14:17 2018

@author: ddo003
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster  import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.externals import joblib
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from threading import Thread
from queue import Queue
import pickle

class ProgressBar(Thread):
    
    def __init__(self, total = "None"):
        Thread.__init__(self)
        self.total = total
        self.working = False
        self.work_done = 0
        self.work_done_percentage = 0
        self.bar_length = 25
    def run(self):
        self.working = True
        start = time.time()
        while self.working:
            t = int(time.time()-start)
                                  
            block = round(self.work_done_percentage * self.bar_length)
            sys.stdout.write("\rTime spent: "+str(t)+" seconds, "+ "["+"#" * block + "-"*(self.bar_length - block) +"] "+str((round(self.work_done_percentage*100, 2)))+"% complete")
            time.sleep(1)
        sys.stdout.write("\n")
        
    def update(self, x = 1):
        self.work_done += x
        self.work_done_percentage = (self.work_done / self.total)
        self.work_left_percentage = 1-self.work_done_percentage
    def stop(self):
        self.working = False

def return_params(df):
    try:  
        d = pd.DataFrame()
#        d["X"] = df.X_zero
#        d["Y"] = df.Y_zero
      
       
        window = 25
        rho_av = []
        drho_av = []
        dphi_av = []
        for ii in range(window, len(df)-window):
            rho_av.append(df.rho5[ii-window: ii+window].mean())
            drho_av.append(df.rho5.diff()[ii-window:ii+window].mean())
            dphi_av.append(df.Turn5[ii-window:ii+window].mean())
        
        spacer = np.zeros(window)
        spacer[:] = np.nan
        for x, c in zip([rho_av, drho_av, dphi_av], ["rho_av", "drho_av", "dphi_av"]):
            x = np.array(x)
            arr = np.hstack([spacer, x, spacer])
            
            d[c] = arr
            
            
        
        return(d.dropna())
    except:
        pass

def classify(dfs, ii, clf):
    try:
        d = return_params(dfs[ii])
        labels = clf.predict(d)
        r = {"index":ii, "labels":labels, "places":d.index}
        
        return(r)
    except:
        r = {"index":ii, "labels":"error", "places":None }
        
        return(r)
    
    
if __name__ == "__main__":
    
        
    print("\nLoading data \n")
    data = pd.read_pickle("Z:\\S13 Projects\Projects active\\Behavioural analysis of ciona larvae\\Behaviour_Data_Frames\\finalDFs\\20180310_alldata.pickle")
#    data = data.loc[data.fps == 30]
    
    n = input("Which Classifier to load? Enter n: ")
    
    clf = joblib.load("CLF"+n)
    
    func = partial(classify, dfs = data.dfs, clf = clf)
    
    to_do = data.loc[data.fps == 30].index
    
    pb = ProgressBar(total = len(to_do))
    pb.start()
    
    
    results = []
    for ii in to_do:
        results.append(classify(data.dfs, ii, clf))
        pb.update()
    pb.stop()
     
    print(" \n \n pickling")
    with open("DataWithLabels.pickle", "wb") as f:
        pickle.dump(results, f)
  

    print("Done")
    
    

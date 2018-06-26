# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:16:42 2017

@author: ddo003
"""

import pandas as pd
import numpy as np
#from sklearn.utils import shuffle
#from sklearn import preprocessing
from sklearn.neighbors import kneighbors_graph
import scipy
#from sklearn.cluster  import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from matplotlib import style
#import pickle
style.use("ggplot")
#from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.externals import joblib
import time
from sklearn.decomposition import PCA
from threading import Thread
import sys, os

def print_waiting():
    global working
    start = time.time()
    while working:
        t = int(time.time()-start)
        sys.stdout.write("\rTime spent: "+str(t)+" seconds")
        time.sleep(1)
    sys.stdout.write("\n")


def centroids(data, model):
    centroids = {}
    for s in set(model.labels_):
        centroid = data[model.labels_ == s].mean(axis =0)
        centroids[s] = centroid
    return centroids

def distance_to_centroids(data, model):
    cent = centroids(data,model)
    distance = []
    print(len(set(model.labels_)))
    for s in set(model.labels_):
        distance.append( np.sum(np.linalg.norm([cent[s], X[model.labels_ == s]])))
    return np.mean(distance)



def agg_clu(n, X, connectivity, fs):
    print("\n n = ", n)
    clf = AgglomerativeClustering(n_clusters=n, linkage='ward',connectivity=connectivity)
    print("\nClustering....")
    clf.fit(X)
    r = (n, clf)
    joblib.dump(r, fs+"_n="+str(n)+".pickle")     
    return n
    
def plot_clusters(clf, X_projected, fs):
    n, clf = clf
    fig = plt.figure(figsize = (20, 20))
    ax = plt.subplot(221)
    ax.scatter(X_projected[:1000000,0],X_projected[:1000000,1], c = clf.labels_[:1000000], cmap = "tab20")
    ax1 = plt.subplot(222)
    ax1.scatter(X_projected[:1000000,0],X_projected[:1000000,2], c = clf.labels_[:1000000], cmap = "tab20")
    ax2 = plt.subplot(223)
    ax2.scatter(X_projected[:1000000,1],X_projected[:1000000,2], c = clf.labels_[:1000000], cmap = "tab20")
#    ax3 = plt.subplot(224)
#    ax3.scatter(X_projected[:1000000,1],X_projected[:1000000,3], c = clf.labels_[:1000000], cmap = "tab20")
    
    fig.savefig(fs+str(n).zfill(2)+".png")
    plt.close()




if __name__ == "__main__":

    working = True
    t = Thread(target = print_waiting)
    t.start()
    print("\nLoading Data \n")
    
    fs = "RDPAvOnly"
    
    X = np.load(fs)



    print("\ncalculating connectivity \n")
    if "connectivity.npz" in os.listdir():
        connectivity = scipy.sparse.load_npz("connectivity.npz")
        print("\nLoaded from drive")
    else:
        connectivity = kneighbors_graph(X, n_neighbors = 4, include_self = False, n_jobs = -1)
        connectivity = 0.5 * (connectivity + connectivity.T)
        scipy.sparse.save_npz("connectivity.npz", connectivity)
    
        print("\nConnectivity calculated \n")
    
    func = partial(agg_clu, X = X, connectivity = connectivity, fs = fs)
#    ns = [3,4,5,6,7,8, 9,10,]
    
    ns = [5,6,7,8,9,10,11,12,13,14,15]
#    clfs = [agg_clu(n, X, connectivity) for n in ns]

    p = Pool(7)
    clfs = p.map(func, ns)
    p.close()
    
    
    print("\nloading Classifiers \n")
    
    
    clfs =  [joblib.load(fs+"_n="+str(n)+".pickle") for n in clfs]
    
    
    print("\nPerforming PCA \n")
    pca = PCA(whiten = True)
    X_projected = pca.fit(X).transform(X)
    
    print("\nPlotting Results \n")
    
    func = partial(plot_clusters, X_projected = X_projected, fs = fs)
    p = Pool(8)
    p.map(func, clfs)
    p.close()
#    
#    for clf in clfs:
#        plot_clusters(clf, X_projected, fs = fs)
    
    
    print("\nPlotting ScreePlot \n")
    ys = [distance_to_centroids(X, x) for _,x in clfs]
    xs = [np.min(ns)+x for x in range(len(ys))]
    
    second_diff = pd.Series(ys).diff().diff()
    first_diff = pd.Series(ys).diff()
    fig = plt.figure(figsize = (20,10))
    ax = plt.subplot(111)
    ax.set_yscale("log")
    ax.plot(xs,ys, label = "Distances to cluster centers", c = "g")
    ax.scatter(xs, ys, c = "k") 
    ax.set_xlabel("N Clusters", fontsize = 20)
    ax.set_ylabel("log of Sum of distance to Centroids", fontsize = 20, color = "g")
    # ax.plot(xs, first_diff, label = "First Differential")
    ax1 = plt.twinx()
    ax1.plot(xs, second_diff, label = "Second Differential", c = "r")
    ax1.set_ylabel("Measure for variance explained", color = "r", fontsize = 20)
    
    plt.legend()
    plt.show()
    fig.savefig("screeplot_log_"+fs+".png")
    
    working = False
    t.join()
    print("Done")
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:41:37 2018

@author: ddo003
"""

#################
#    IMPORTS    #
#################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#switch off warnings for chained indexing
pd.options.mode.chained_assignment = None  # default='warn'
import os
import tkinter as tk
from tkinter import filedialog

from scipy import signal
import sys
from sklearn.preprocessing import scale
from functools import partial

from multiprocessing import Pool, cpu_count

# Well calibration factor (um per pixel)
calibration_factor = 11.56

#define names of columns that are made by Toxtrac
xs = "Pos. X (mm)"
ys = "Pos. Y (mm)"
dftime = "Time (sec)"


########################
#    FUNCTIONS USED    #
########################

def get_path():
    """opens tkinter based filedialog"""
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.askdirectory()
#    os.chdir(path)
    return path

def cart2pol(x, y):
    """converts cartesian vectors to polar vectors"""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return [rho, phi]

def calculate_polar(x, y):
    """creates a list of polar vectors from a list of cartesian vectors"""
    rho = []
    phi = []
    for ii in range(len(x)):
        r, p = cart2pol(x[ii], y[ii])
        rho.append(r)
        phi.append(p)
    return rho, phi

def calculate_distance(x,y):
    """calculates euclidian distances when given columns with x and y coordinates"""
    dist = [None]    
    for ii in range(1,len(x)):
        x1 = x[ii]
        y1 = y[ii]
        x2 = x[ii-1]
        y2 = y[ii-1]
        distance = np.sqrt(((x2-x1)**2)+(y2-y1)**2)
        dist.append(distance)
    return pd.Series(dist)

def calculate_displacement(x,y):
    """calculates euclidian diplacement from starting coordinates when given columns with x and y coordinates"""
    dist = [None]  
    x1 = x[0]
    y1 = y[0]
    for ii in range(1,len(x)):
        
        x2 = x[ii-1]
        y2 = y[ii-1]
        distance = np.sqrt(((x2-x1)**2)+(y2-y1)**2)
        dist.append(distance)
    return pd.Series(dist)

def pythagoras(x,y):
    """simple pythagoras to calcuate distance from center for thigmo"""
    hypot = []    
    for ii in range(len(x)):
        hypotenuse= np.sqrt(x[ii]**2 + y[ii]**2) 
        hypot.append(hypotenuse)
    return hypot

def calculate_vectors(U, V):
    """Calculate vectors from the supplied coordinate lists U and V."""
    vectors = [[0,0]]
    for jj in range(1,len(U)):
        u = U[jj]-U[jj-1]
        v = V[jj]-V[jj-1]
        vectors.append([u,v])
    return vectors    

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
        Mx[:,ii] = Mx[:,ii] - np.nanmean(Mx[:,ii])
        My[:,ii] = My[:,ii] - np.nanmean(My[:,ii])
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


def remove_outliers(group, stds, replacement):
    """replaces outliers in group that are over stds with replacement"""
    group[np.abs(group - group.mean()) > stds * group.std()] = replacement
    return group



###############################
#    MAIN WORKING FUNCTION    #
###############################

def calculate_df(file, framerate, path, savedir):
    """Takes a path to a tracking file"""
    try:
        
        specs = file.split("-") 

        #Get well-centers by retrieving position in pixels from filename and converting to um with our accurate calibration
        well_x = int(specs[7])*calibration_factor
        well_y = int(specs[8])*calibration_factor
 
        
        file_path = os.path.join(path, file)
        d = pd.read_csv(file_path, delimiter = "\t")
        
        df = pd.DataFrame({"Frame":range(d.Frame.iloc[-1])})
        
        df = pd.merge(df, d, how = "outer", on = "Frame")
        
#        df["Frame_zero"] = pd.Series([x for x in range(len(df))])
        
        df["X"] = df["X"]*calibration_factor #convert X_locations from pixel to um
        df["Y"] = df["Y"]*calibration_factor #convert Y_locations from pixel to um
        
        
        df["Distance"] = calculate_distance(df.X, df.Y) #calculate distances between coordinates in trace
        
        
        #####################################################################################################
        #   THE FOLLOWING BLOCK MUST BE RETHOUGHT: JUMP REMOVAL/FILTERING
        #####################################################################################################
        
        df["Dist_corr"] = df["Distance"] #set up a column to get corrected distances into
        df["Dist_corr"] = remove_outliers(df["Dist_corr"], 3, "out") #replace values over 3 stds with "out"
        df["X_corr"] = df["X"] #set up columns to dump x_corrected and y_corrected locations
        df["Y_corr"] = df["Y"]
        #set all distances and locations in the corrected columns to nan if they are outliers
        df.loc[df["Dist_corr"] == "out", "X_corr"] = np.nan
        df.loc[df["Dist_corr"] == "out", "Y_corr"] = np.nan
        df.loc[df["Dist_corr"] == "out", "Dist_corr"] = np.nan
        
        #refill the nan-values with linear interpolations
        for c in ["Dist_corr", "X_corr", "Y_corr"]:
            df[c+"_interpol"] = df[c].interpolate(method = "linear")
        
        #df["Dist_corr"+i] = remove_outliers(df["Dist_corr"+i], 3, np.nan)  
        ########################################################################################################
        
        
        #center traces around well-center
        df["X_zero"] = (df["X_corr"]-well_x)
        df["Y_zero"] = (df["Y_corr"]-well_y)
        
        df["Displacement"] = calculate_displacement(df.X_zero, df.Y_zero)
        df["Speed"] = df["Dist_corr"] / (1/framerate)
        
#        #calculate corrected speeds IS THIS STILL NEEDED?
#        df["Speed_corr"] = remove_outliers(df["Speed"], 3, np.nan)
        
        #Should perhaps be delta-Rho? Should be the same thing. 
        df["Acceleration"] = df["Speed"].diff()
        
        
        #Calculate velocities
        for c in ["X", "Y"]:
            df["V"+c] = df[c+"_corr"].diff()
            df["V"+c+"5"] = df[c+"_corr"].diff(periods = 5)
        df["rho"], df["phi"] = calculate_polar(df["VX"], df["VY"])
        df["rho5"], df["phi5"] = calculate_polar(df["VX5"],df["VY5"])
        
        #set outliers to nan again:
        for c in ["rho", "phi"]:
            df[c][df["Dist_corr"].isnull()] = np.nan
            df[c+"5"][df["Dist_corr"].isnull()] = np.nan
        
                
        
        df["Turn"] = df["phi"].diff()
        df["Turn"][df["Turn"] > np.pi] -= 2*np.pi
        df["Turn"][df["Turn"] < - np.pi] += 2*np.pi
        
        df["Turn5"] = df["phi5"].diff()
        df["Turn5"][df["Turn5"] > np.pi] -= 2*np.pi
        df["Turn5"][df["Turn5"] < - np.pi] += 2*np.pi
             
       
       
        b, a = signal.butter(1, (1/(framerate/2)), 'low') #create butterworth filter with cutof frequency 1 Hz  
        xs = df["Frame"][df["Speed"].notnull()]
        df["Filtered_Speed"] = df["Speed"]
        filtered_speed = signal.filtfilt(b,a,df["Speed"].dropna())
        df["Filtered_Speed"][xs] = filtered_speed
        
         ###distance from center //TEST!?
        df["From_center"]=  pythagoras(df["X_zero"],df["Y_zero"])
        
        #Generating Truth-values for thigmo. 307 pixels is chosen to get inner and outer surface area equal
        df["Thigmo"] = df.From_center > (307*calibration_factor)
        ###
        
        window = framerate * 3
        M = obtain_M(df.X_zero, df.Y_zero, window = window)
        lH,H = get_H(M)
        df["lH"] = np.hstack((lH, np.array([np.nan]*window)))
        
        
        
        
        save_path = os.path.join(savedir, file)
#        df.to_csv(save_path, sep = "\t")
        return([save_path, df])
        
    except:
        save_path = os.path.join(savedir, file)
        return([save_path, "error"])


################################
# LOADING DATA AND SETTING UP  #
################################


if __name__ == "__main__":

    #Ask user for the path to the data directory and the saving directory
    path = get_path()
    savedir = get_path()
    
    #Framerate of the video is extracted from the path: If 10fps is not mentioned
    #in the path the program will assume 30fps, else 10.
    framerate = 30 if "10fps" not in path.lower() else 10
    
    #confirm with user that framerate is correct
    contin = input("Detected framerate is %s fps, is this correct? [y/n] :" %str(framerate))
    
    if contin == "y":
        pass
    else:
        framerate = int(input("Please enter the correct framerate: "))
    
    #get all files in the datadirectory
    print("Initializing Files")
    files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]    

    func = partial(calculate_df, path = path, savedir = savedir, framerate = framerate)
    print("Starting analysis")
    print("Remember: Patience is a virtue")
    pool = Pool(cpu_count() -1)
    
    results = pool.map(func, files)
    
    pool.close()
    print("Done, saving results....")
    for r in results:
        if type(r[1]) == str:
            print( "Error for "+r[0])
        else:
#            print("Pickling "+ r[0])
            r[1].to_pickle(r[0])
    
    print("***All is well***")










































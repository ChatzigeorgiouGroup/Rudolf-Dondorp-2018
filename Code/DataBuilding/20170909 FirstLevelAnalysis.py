# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:50:48 2017

@author: ddo003
"""

import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from centerfinding import find_center

import multiprocessing
xs = "Pos. X (mm)"
ys = "Pos. Y (mm)"
dftime = "Time (sec)"

import sys


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




#loop to go throughg all the tracking files, break down the paths and extract info from the yielded info
def process_tracking_files(f, savepath):
    """takes the paths from var tracking_files, gathers all required info and saves a file per trace in the savedir"""
    global errorlog
    try:
        #split path to list
        path_contents = f.split("\\")
        #vid name is same as foldername one dir up
        vid = path_contents[-2]
        #date is first dir in path   
        date = path_contents[1][:8]
        
        #get state of video/analysis from metadatafile
        state = metadata.state[metadata.Name == vid].values[0]
       
        #get hatching info for videos that need it from hatching data file and calculate true age
        if int(date) in hatching_info.date.values:
            hatching = hatching_info["hatching (hpf)"][hatching_info.date == int(date)].values[0]
        else:
            hatching = 42  
        age = vid[2:4]
        true_age = str(int(age)-int(hatching)).zfill(2)
        
        #get crowdsize and chorionation state from vid name
        crowd = vid[0]
        chor = vid[1]
        
        #do centerfinding on the appropriate video, and correct the found center for the arena position in toxtrac
        vid_belonging_to_file = [x for x in video_file_paths if date in x and vid in x][0]
        corners = corners_per_video[vid]
        center = find_center(vid_belonging_to_file)
        center_x = str(center[0]-int(corners[0]))
        center_y = str(center[1]-int(corners[1]))
        
        #get lightstim_treatment out of vidnames
        if "green" in vid.lower():
            light = "green"
        elif "red" in vid.lower():
            light = "red"
        elif "blue" in vid.lower():
            light = "blue"
        else:
            light = "none"
        
        #open the tracking file as pd dataframe
        data = pd.read_csv(f, delimiter = "\t", names = ["Frame", "Arena", "Track", "X","Y","Label"])
        
        #extract 
        tracks = []
        for t in data.Track:
            if t not in tracks:
                tracks.append(t)
        
        if len(tracks) > int(crowd):
            print("Tracecount exceeded animal count for "+f)
        else:
            for t in tracks:
                df = pd.DataFrame()
                df["Frame"] = data["Frame"].loc[data.Track == t]
                df["X"] = data["X"].loc[data.Track == t]
                df["Y"] = data["Y"].loc[data.Track == t]

                savename = date+"-"+vid+"-"+state+"-"+chor+"-"+true_age+"-"+crowd+"-"+light+"-"+center_x+"-"+center_y+"-"+str(t)+".txt"
                save_name_path = os.path.join(savepath, savename)
#                df.to_csv(save_name_path, sep = "\t")
                return( [save_name_path, df] )
    except:
        print("mysterious error for "+f)
        errorlog = errorlog + date +"\t"+ vid + "\n"
        return( [save_name_path, "error"] )

#############################
# THIS CAN BE MULTIPROCESSED#
#############################
#for f in tracking_files:
#    process_tracking_files(f)

########################
# MULTIPROCESSING CODE #
########################
if __name__ == "__main__":
    
        #select mother directory
    path = get_path(name = "Choose data directory")
    metadata_file_path = get_filename(name = "Choose Metadata File")
    hatching_info_path = get_filename(name = "Choose HatchingInfo File")
    savepath = get_path(name = "Choose output directory")
    
    #get metadata and hatching data from the excel files made by Louise
    metadata = pd.read_excel(metadata_file_path)
    hatching_info = pd.read_excel(hatching_info_path)
    
    #Get paths to all tracking_01.txt files that contain our coordinates for our traces present in the mother directory 
    tracking_files = [os.path.join(root, name) for root,dirs,files in os.walk(path) for name in files if "tracking" in name.lower() and "realspace" not in name.lower()] 
        
    #Get paths and filenames for all .avi files present in the mother directory
    video_file_paths  = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files if name.endswith(".avi")]    
    video_file_names  = [name for root, dirs, files in os.walk(path) for name in files if name.endswith(".avi")]
    
    #Get paths to all relevant arena files present in mother directory
    #arena_paths = [os.path.join(root, name) for root, dirs,files in os.walk(path) for name in files if "arena" in name.lower() and "names" not in name.lower() and "active" not in name.lower()]    
    arena_roots = [root for root, dirs,files in os.walk(path) for name in files if "arena" in name.lower() and "names" not in name.lower() and "active" not in name.lower()]
    
    #create dictonary where arena corners are stored per video name
    corners_per_video = {}
    for root in arena_roots:
        try:
            arena_file = [os.path.join(root,name) for name in os.listdir(root) if "arena" in name.lower() and "names" not in name.lower() and "active" not in name.lower() ][0]
            with open(arena_file, "r") as f:
                arena_specs = f.readlines()[1].strip("\n").split("\t")
            arena_corners = arena_specs[:2]
                
            video_names_in_root = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x))]
            for video_name in video_names_in_root:
                corners_per_video[video_name] = arena_corners
        except:
            print("Could not find proper arena info for "+ root)
    
    
    errorlog = ""
    
    
    
    print("starting to work")
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus-1)
    
    results = pool.map(process_tracking_files, tracking_files)
    
    pool.join()
    pool.close()
    print("Nearly there...!")
    
    for r in results:
        if type(r[1]) == str:
            print("Error for "+r[0])
        else:
#            print("Pickling "+ r[0])
            r[1].to_csv(r[0], delimiter = "\t")
    
    
#############################
#   END OF MULTIPROCESSING  #
#############################

    
#errorlog_path = os.path.join(path, "errorlog.txt")
#with  open(errorlog_path,"w") as e:
#    e.write(errorlog)
#print("All is well")

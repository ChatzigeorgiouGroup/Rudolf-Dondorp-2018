import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from multiprocessing import Pool

data = pd.read_pickle("D:\\20180118_alldata.pickle")
data = data.loc[(data.state == "good") & (data.fps == 30) & (data.lazy != True)]
dfs = data.dfs
data = None


def return_params(ii):
    try:
        global dfs
        df = dfs[ii]
        d = pd.DataFrame()
        d["X"] = df.X_zero
        d["Y"] = df.Y_zero
      
       
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
            
            
        
        return(d)
    except:
        pass

if __name__ == "__main__":
    p = Pool(8)
    dataframes = p.map(return_params, [x for x in dfs.index])
    p.close()
    dataframes = pd.concat(dataframes, ignore_index = True)
    dataframes.drop(["X","Y"], axis = 1, inplace = True)
    dataframes.dropna(inplace = True)
    X = dataframes.values
    X = shuffle(X)
    X.dump("RDPAvOnly")
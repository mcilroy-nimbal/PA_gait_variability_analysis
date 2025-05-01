''' this
'''

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_sec,read_orig_fix_clean_demo, read_demo_ondri_data
from variability_analysis_functions import alpha_gini_index
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import datetime
import openpyxl
from scipy.stats import gaussian_kde

#set up paths
root = 'W:'
#check - but use this one - \prd\nimbalwear\OND09
path1 = root+'\\prd\\NiMBaLWEAR\\OND09\\analytics\\'

nimbal_drive = 'O:'
paper_path =  '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
log_out_path = nimbal_drive + paper_path + 'Log_files\\'
summary_path = nimbal_drive + paper_path + 'Summary_data\\'

nw_path = 'nonwear\\daily_cropped\\'
bout_path = 'gait\\bouts\\'
step_path = 'gait\\steps\\'
daily_path = 'gait\\daily\\'
sptw_path = 'sleep\\sptw\\'


###########################################
#read in the cleaned data file for the HANNDS methods paper
nimbal_dr = 'o:'
new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
demodata = read_demo_ondri_data(nimbal_dr, new_path)

#gini runs
bouts_not_density = False #if using bout data TRUE else False for density
bout_step = False # if using n steps in bout - False if using duration
set_xmin = -1 #-1 if no setting of XMIN

source = 'density'
type = 'min'
dur_type ='1min'

########################################################
# loop through each eligible subject
# File the time series in a paper specific forlder?
#extract on a few variabels from the demo
demodata = demodata[['SUBJECT','COHORT','AGE', 'EMPLOY_STATUS']]
demodata['gini', 'alpha', 'xmin','fits', 'npts'] = None


for index, row in demodata.iterrows():
    print(f'\rFind subjs - Progress: {index}' + ' of ' + str(len(demodata)), end='', flush=True)
    #remove the underscoe that is in the subject code from the demodata file

    parts = row['SUBJECT'].split('_', 2)  # Split into at most 3 parts
    if len(parts) == 3:
        subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
        visit = '01'

        #FIND ALL THE DENSITY FIELS THAT MACTH
        #read in density and append to one array
        #use that data for gini
        try:
             #subejct has hyphen between OND)( and subj in this file name
             density = pd.read_csv(summary_path+'density\\'+ subject + '_' + visit + '_'+ dur_type+'_density.csv')

        except:
             #log_file.write('Steps file not found - Subject: '+subject+ '\n')
             continue

        #loop through
        fig = plt.figure(figsize=(12, 6))
        kde_x = np.linspace(0, 75, 100)
        density = density[density != 0]

        for i, col in enumerate(density.columns[1:]):
            #data = data[data != 0]
            #ax.plot(density.index, [i] * len(density), density[col])
            signal = density[col].values

            signal = signal[~np.isnan(signal)]


            kde = gaussian_kde(signal, bw_method=0.1)
            plt.plot(kde_x, kde(kde_x))

        plt.show()

        #mergae all the data columns to one array and remove zeros
        #data = density.to_numpy().flatten()
        #data = data[data != 0]





''' this
'''

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import (wake_sleep, bout_bins, steps_by_day, step_density_sec,read_orig_fix_clean_demo,
                       read_demo_ondri_data, summary_density_bins, read_demo_data)
from variability_analysis_functions import alpha_gini_index
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import datetime
import openpyxl
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import nolds


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

#study = 'OND09'
study = 'SA-PR01'

demodata = read_demo_data(study)

#gini runs
bouts_not_density = False #if using bout data TRUE else False for density
bout_step = False # if using n steps in bout - False if using duration
set_xmin = -1 #-1 if no setting of XMIN

source = 'density'
type = 'min'
dur_type ='60sec'
#dur_type ='15sec'

########################################################
# loop through each eligible subject
# File the time series in a paper specific forlder?
#extract on a few variabels from the demo
demodata = demodata[['SUBJECT','COHORT','AGE', 'EMPLOY_STATUS']]
demodata['gini', 'alpha', 'xmin','fits', 'npts'] = None

density_header = ['SUBJECT','COHORT','AGE', 'EMPLOY_STATUS', 'day', 'n_steps', 'DFA-alpha','Sample-entropy', 'n_total', 'zero_total', 'vlow_total', 'low_total', 'med_total', 'high_total', 'bout_3min', 'bout_10mn']
density_sum = pd.DataFrame(columns=density_header)

#with PdfPages(summary_path+'\\all_total_1min.pdf') as pdf:

kde_x = np.linspace(0, 80, 100)
kdes = []
ages = []
cohorts = []

for index, row in demodata.iterrows():
    print(f'\rFind subjs - Progress: {index}' + ' of ' + str(len(demodata)) + '   Subj: '+str(row['SUBJECT']), end='', flush=True)
    #remove the underscoe that is in the subject code from the demodata file

    if study == 'OND09':
        parts = row['SUBJECT'].split('_', 2)  # Split into at most 3 parts
        if len(parts) == 3:
            subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
    elif study == 'SA-PR01':
        subject = 'SA-PR01_' + row['SUBJECT']
    visit = '01'

    #FIND ALL THE DENSITY FIELS THAT MACTH
    #read in density and append to one array

    try:
        #subejct has hyphen between OND)( and subj in this file name
        density = pd.read_csv(summary_path+'density\\'+ subject + '_' + visit + '_'+ dur_type+'_density.csv')

    except:
        #log_file.write('Steps file not found - Subject: '+subject+ '\n')
        continue

    #loop through
    density = density.iloc[:,1:]

    #fig = plt.figure(figsize=(12, 6))

    # by day
    signal_long = []
    for i, col in enumerate(density.columns):

        curr_day = density.loc[0,col]
        signal = density.loc[1:,col].values
        signal = list(map(int, signal))
        signal = np.array(signal)
        total = sum(signal)
        #signal = signal[~np.isnan(signal)]
        #sum densoity across bins (each day)
        sum_density = summary_density_bins(signal)
        sum_density = [int(v) for v in sum_density]
        if len(signal) > 1:
            #DFA for the day acvitites
            alpha = float(nolds.dfa(signal))

            #Sample entropy
            apen = nolds.sampen(signal)
        else:
            alpha = None
            apen = None

        add_row = [subject,  row['COHORT'], row['AGE'], row['EMPLOY_STATUS'], curr_day, float(total),
                  float(round(alpha,4)), float(round(apen,4))] + sum_density
        density_sum.loc[len(density_sum)] = add_row
        signal_long.extend(signal)

    #mergae all the data columns to one array and remove zeros
    #density = density.iloc[1:,1:]

    #data = density.to_numpy().flatten(order='F')
    signal_long = list(map(int, signal_long))
    signal_long = np.array(signal_long)
    total = sum(signal_long)
    if index != 47:
        # DFA for the ALL data
        alpha = float(nolds.dfa(signal_long))
        # Sample entropy
        apen = nolds.sampen(signal_long)

        #summaize total across bin sizes
        sum_density = summary_density_bins(signal_long)
        sum_density = [int(v) for v in sum_density]
        add_row = [subject, row['COHORT'], row['AGE'], row['EMPLOY_STATUS'], 'all', float(total),
              float(round(alpha, 4)), float(round(apen, 4))] + sum_density
        density_sum.loc[len(density_sum)] = add_row

#write sumamry data to
summary_path = nimbal_drive + paper_path + 'Summary_data\\'
density_sum.to_csv(summary_path+study+'_density_summary_v3.csv', index=False)


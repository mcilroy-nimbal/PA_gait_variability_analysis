#THIS CODE: analysis the data produced in STEP 1
#Bins file
#Density files

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_1min,read_orig_clean_demo
import numpy as np
import seaborn as sns
import datetime
import openpyxl

#set up paths
root = 'W:'
nimbal_drive = 'O:'
paper_path =  '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
log_out_path = nimbal_drive + paper_path + 'Log_files\\'
summary_path = nimbal_drive + paper_path + 'Summary_data\\'

###########################################
#read in the cleaned data file for the HANNDS methods paper
nimbal_dr = 'o:'
new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
#Import data files - use this if file already created
demodata = pd.read_csv(nimbal_dr+new_path+"OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")

'''
#Gait bout bin analysis
#read summary bin file
bouts = pd.read_csv(summary_path + 'steps_daily_bins.csv')

sum1 = []
for subj in bouts['subj'].unique():
    sub_set = bouts[bouts['subj'] == subj]
    n_days = len(sub_set)
    tot_steps = sub_set['total'].sum()
    tot_steps_day = tot_steps / n_days
    unbouted = sub_set['<_3'].sum()
    short = sub_set['<_5'].sum() + sub_set['<_10'].sum()
    medium = sub_set['<_20'].sum() + sub_set['<_50'].sum()
    long = sub_set['<_100'].sum() + sub_set['<_300'].sum() + sub_set['>_300'].sum()
    row = {'subj': subj, 'ndays':n_days, 'unbout': unbouted, 'short': short,
           'medium': medium, 'long': long, 'total':tot_steps}
    sum1.append(row)

sum_bouts = pd.DataFrame(sum1, columns=['subj','ndays','unbout','short','medium','long','total'])
x = sum_bouts['total']/sum_bouts['ndays']
unbout = sum_bouts['unbout']/sum_bouts['ndays']
short = sum_bouts['short']/sum_bouts['ndays']
med = sum_bouts['medium']/sum_bouts['ndays']
long = sum_bouts['long']/sum_bouts['ndays']
plt.scatter(x, unbout, color='red')
plt.scatter(x, short, color='orange')
plt.scatter(x, med, color='blue')
plt.scatter(x, long, color='green')
plt.show()

print ('done')
'''

# bout density data
data_path = summary_path+'density\\'
files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
counter = 0
bin_width = 5
# Compute bin edges
bins = np.arange(start=-bin_width, stop=100, step=bin_width)
bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
columns = ['subj','day'] + bin_labels
hist = pd.DataFrame(columns=columns)
step = -50
for index, file in enumerate(files):
    print(f'\rSubj #: {index}' + ' of ' + str(len(files)), end='', flush=True)
    subj_density = pd.read_csv(data_path+file)
    del subj_density[subj_density.columns[0]]

    # Create histogram
    for day, col in enumerate(subj_density.columns):
        # Bin the data
        binned = pd.cut(subj_density[col], bins=bins, right = True)
        # Count the number of observations in each bin
        counts = binned.value_counts().sort_index()
        row = counts.values
        plt.plot(bin_labels, row)
        row = np.concatenate(([index,day],row))
        hist.loc[len(hist)] = row

plt.show()
hist.to_csv(summary_path+'freq_step_per_min.csv')
print ()
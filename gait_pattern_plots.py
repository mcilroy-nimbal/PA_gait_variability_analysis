import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

################## Read to plot
#out_path ='O:\\Student_Projects\\gait_pattern_paper_feb2024\\'
#out_file = 'stats1_feb25.csv'
#stats = pd.read_csv(out_path+out_file)
#sns.jointplot(data=stats, x='total_med', y='no_sleep_med')
#sns.jointplot(data=stats, x='no_sleep_med', y='sleep_med')
#plt.scatter(stats['total_med'], stats['no_sleep_med'])
#plt.errorbar(stats['total_med'], stats['wake_med'], xerr=stats['total_std'], yerr=stats['wake_std'])
#plt.show()


#######################################################
# sumamrizing sleep and non sleep bouts
out_path ='O:\\Student_Projects\\gait_pattern_paper_feb2024\\'

sleep_all = pd.DataFrame(columns=['<3', '3-5', '5-10', '10-20', '20-30', '30-50', '50-100', '>100'])
file = 'sleep_bouts_feb25.csv'
bouts = pd.read_csv(out_path+file)
#total_steps = bouts.iloc[:,3:].sum()
#bouts['total'] = total_steps
#bouts_sorted = bouts.sort_values(by='total', ascending=True)

subj = bouts['subj'].unique()
n_subjs = len(subj)
for i in subj:
    bout_sum = [0]*8
    temp = bouts[bouts['subj'] == i]
    ndays = len(temp)
    total_steps = temp.iloc[:,3:].sum().sum()

    for i in range(8):
        bout_sum[i] = temp.iloc[:,i+3].median()
        #bout_sum[i] = 100*(temp.iloc[:, i + 3].sum())/total_steps
    sleep_all.loc[len(sleep_all)] = bout_sum

###################################
wake_all = pd.DataFrame(columns=['<3', '3-5', '5-10', '10-20', '20-50', '50-100', '100-300', '>300'])
file = 'wake_bouts_feb25.csv'
bouts = pd.read_csv(out_path+file)
subj = bouts['subj'].unique()
n_subjs = len(subj)
for i in subj:
    bout_sum = [0]*8
    temp = bouts[bouts['subj'] == i]
    ndays = len(temp)
    total_steps = temp.iloc[:, 3:].sum().sum()
    for i in range(8):
        bout_sum[i] = temp.iloc[:,i+3].median()
        #bout_sum[i] = 100*(temp.iloc[:, i + 3].sum()) / total_steps

    wake_all.loc[len(wake_all)] = bout_sum






ax = plt.figure(figsize=(14, 5))
ct = 0
for col in wake_all.columns[:]:
    ax = sns.violinplot(data=wake_all, y=col, x=ct)

    ct += 2
#ct = 1
#for col in sleep_all.columns[:]:
#    ax = sns.violinplot(data=sleep_all, y=col, x=ct)
#    ct += 2
#ax.set_xticklabels(['<3', '3-5', '5-10', '10-20', '20-50', '50-100', '100-300', '>300',
#                    '<3', '3-5', '5-10', '10-20', '20-50', '50-100', '100-300', '>300'])
plt.show()

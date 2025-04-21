''' this STEP3 runs the various variability analysis across eligible subjects and writes data to file

'''

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_1min,read_orig_clean_demo
from variability_analysis_functions import alpha_gini_index
import numpy as np
import seaborn as sns
import datetime
import openpyxl

gini = True

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
#this woudl read in the elegible subejcts with demogrpahic data
#demodata = read_orig_clean_demo()
#Import data files - use this if file already created
demodata = pd.read_csv(nimbal_dr+new_path+"OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")

########################################################
# loop through each eligible subject
# File the time series in a paper specific forlder?
master_subj_list = []
for i, subject in enumerate(demodata['SUBJECT']):
    print(f'\rFind subjs - Progress: {i}' + ' of ' + str(len(demodata)), end='', flush=True)
    #remove the underscoe that is in the subject code from the demodata file
    parts = subject.split('_', 2)  # Split into at most 3 parts
    if len(parts) == 3:
        subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
    master_subj_list.append(subject)


if gini:

    gini_steps = []
    gini_dur = []
    #PART A - loop and do bin counts
    for j, subject in enumerate(master_subj_list):
        print(f'\rSubject - Progress: {j}' + ' of ' + str(len(master_subj_list)), end='/n', flush=True)
        visit = '01'
        #get step data for subject
        try:
            bouts = pd.read_csv(path1 + bout_path + subject + '_' + visit + '_GAIT_BOUTS.csv')
            ''' study_code, subject_id, coll_id, gait_bout_num, start_time, end_time, step_count'''
        except:
            #log_file.write('Steps file not found - Subject: '+subject+ '\n')
            continue
        #by step number
        data = bouts['step_count']
        g_steps, alpha_steps, xmin_staps, n_steps = alpha_gini_index (data, plot=False)
        gini_steps.append(g_steps)

        #by duration
        bouts['start'] = pd.to_datetime(bouts['start_time'])
        bouts['end'] = pd.to_datetime(bouts['end_time'])

        # Calculate difference in seconds
        bouts['duration'] = (bouts['end'] - bouts['start']).dt.total_seconds()
        g_dur, alpha_dur, xmin_dur, n_dur = alpha_gini_index(bouts['duration'], plot=False)
        gini_dur.append(g_dur)

    plt.figure(figsize=(8, 5))
    plt.hist(gini_steps, bins=10, alpha=0.6, label='Step #', color='blue', edgecolor='black')
    plt.hist(gini_dur, bins=6, alpha=0.6, label='Duration', color='orange', edgecolor='black')
    plt.title('Frequency Distribution of Gait Bout Durations')
    plt.xlabel('Gini Index')
    plt.ylabel('Frequency')
    plt.show()


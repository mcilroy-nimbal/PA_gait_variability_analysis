import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Functions import (wake_sleep, steps_by_day, step_density_sec,
                       read_demo_ondri_data, read_demo_data, stride_time_interval,
                       create_bin_density_files, select_subjects)
from Gait_bout_basic_anal_graphs import (plot_stride_bouts_histogram, calc_basic_stride_bouts_stats, bouts_SML)
import numpy as np
import seaborn as sns
import datetime
import openpyxl
import warnings
warnings.filterwarnings("ignore")

#set up file paths
study = 'OND09'
root = 'W:'
nimbal_drive ='O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
demo_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'

bin_list_steps = [5, 10, 25, 50, 100, 300]
bin_width_time = [5, 10, 30, 60, 180, 600]

#which subjects?
master_subj_list = select_subjects(nimbal_drive, study)
#master_subj_list = ['OND09_SBH0038']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

#create summary data files
create = False
if create:
    time_window ='1010' #'24hr' #'1010'  'wake'

    create_bin_density_files(time_window, study, root, nimbal_drive, paper_path, master_subj_list,
                             bin_list_steps, bin_width_time)


calc_basic_stats = False
if calc_basic_stats:
    #select specific subjects from the study group
    #select ONDRI controls
    path = nimbal_drive + demo_path
    demodata = read_demo_ondri_data(path)
    subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
    group_name = 'Control'
    window = '1010'

    step_vs_dur = True
    #calcualte medians and std for each bout and clustred bouts
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_list, group_name)

    step_vs_dur = False #run duraiton file
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_list, group_name)


    #SML_median, SML_std, SML_pct_median, SML_pct_std = bouts_SML(nimbal_drive, study, window, paper_path, subj_list)

plot = True
if plot:
    path = nimbal_drive + demo_path
    group_name = 'Control'
    study = 'OND09'

    path_24hr = nimbal_drive + paper_path + 'Summary_data\\' + study + '_24hr_' + group_name + '_bout_duration_'
    subj_24hr = pd.read_csv(path_24hr +'_subj_stats.csv' )
    subj_pct_24hr = pd.read_csv(path_24hr + '_pct_subj_stats.csv')

    path_1010 = nimbal_drive + paper_path + 'Summary_data\\' + study + '_1010_' + group_name + '_bout_duration_'
    subj_1010 = pd.read_csv(path_1010 +'_subj_stats.csv' )
    subj_pct_1010 = pd.read_csv(path_1010 + '_pct_subj_stats.csv')




    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame({'Value': pd.concat([subj_24hr['window_total_strides.1'], subj_1010['window_total_strides.1']], ignore_index=True),
        'Feature': ['24 hr'] * len(subj_24hr) + ['10AM-10PM'] * len(subj_1010)})

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Feature', y='Value', data=plot_df, inner=None, palette='Set2')
    sns.stripplot(x='Feature', y='Value', data=plot_df, jitter=True, color='black', size=4)
    plt.ylim(0, 20000)  # Set min and max range
    #plt.yticks(range(0, 7000, 1000))  # Set ticks every 5 units

    plt.title('Distribution of 24hr versus 10-10')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()


    #Figure 1 - spread of total strides for each window (24hr, 1010 and eventually wake)



    plot_stride_bouts_histogram(nstride_all_median, nstride_all_std, nstride_pct_all_median, nstride_pct_all_std,
                                totalTF=False)
    print ('pause')

    #bouts_SML(nimbal_drive, study, window, paper_path, subj_list)
    print ('pause')

#plotting


#analysis for paper
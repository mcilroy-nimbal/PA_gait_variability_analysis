import pandas as pd
import glob
import os
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


plot_bouts = True
if plot_bouts:
    #select specific subjects from the study group
    #select ONDRI controls
    path = nimbal_drive + demo_path
    demodata = read_demo_ondri_data(path)
    subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
    window = '24hr'

    #calcualte medians and std for each bout and clustred bouts
    nstride_all_median, nstride_all_std, nstride_pct_all_median, nstride_pct_all_std \
        = calc_basic_stride_bouts_stats(nimbal_drive, study, window, paper_path, subj_list)

    #SML_median, SML_std, SML_pct_median, SML_pct_std = bouts_SML(nimbal_drive, study, window, paper_path, subj_list)

    plot_stride_bouts_histogram(nstride_all_median, nstride_all_std, nstride_pct_all_median, nstride_pct_all_std,
                                totalTF=False)
    print ('pause')

    #bouts_SML(nimbal_drive, study, window, paper_path, subj_list)
    print ('pause')

#plotting


#analysis for paper
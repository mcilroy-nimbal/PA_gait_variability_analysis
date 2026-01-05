import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from Functions import (wake_sleep, steps_by_day, step_density_sec,
                       read_demo_ondri_data, read_demo_data, stride_time_interval,
                       create_bin_files, create_density_files, select_subjects,
                       all_bouts_histogram, create_table, alpha_gini_bouts)
from Gait_bout_basic_anal_graphs import (calc_basic_stride_bouts_stats, bouts_SML)

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

#values for density analysis
window_size = 15
step_size = 1

#which subjects?
master_subj_list = select_subjects(nimbal_drive, study)
#master_subj_list = ['OND09_SBH0006']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

#select specific subjects from the study group
#select ONDRI
group_name = ['Control', 'PD', 'ADMCI']

study = 'OND09'
path = nimbal_drive + demo_path
demodata = read_demo_ondri_data(path)
#demogarphci details by group
#AGE,SEX,MRTL_STATUS,EMPLOY_STATUS, LIVING_CIRCUM
#data = demodata
#categ, cont = create_table(data, ['AGE'], ['SEX','MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])

for i in range(3):
    if i == 0:
        label = 'Community Dwelling'
    elif i == 1:
        label = 'PD'
    elif i == 2:
        label = 'AD_MCI'
    #data = demodata[demodata['COHORT'] == label]
    #categ, cont = create_table(data, ['AGE'], ['SEX','MRTL_STATUS','EMPLOY_STATUS', 'LIVING_CIRCUM'])
    #print (group_name[i], categ, cont)

#subject lists
list1 = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
list2 = demodata[demodata['COHORT'] == 'PD']['SUBJECT']
list3 = demodata[demodata['COHORT'] == 'AD/MCI']['SUBJECT']
subj_lists = [list1, list2, list3]

#which grousp to run data on
#groups  - 0 Control, 1 PD, 2 ADMCI
group = 0

#read the bouts data to get step totals for the analysis
path = nimbal_drive + demo_path
path_24hr = nimbal_drive + paper_path + 'Summary_data\\' + study + '_24hr_' + group_name[group] + '_bout_duration_'
subj_24hr = pd.read_csv(path_24hr +'_subj_stats.csv', header=[0, 1], skiprows=[2])
subj_pct_24hr = pd.read_csv(path_24hr + '_pct_subj_stats.csv', header=[0, 1], skiprows=[2])


create_density = True
create_stride_time = False
calc_basic_stats = False
density_graph = True
window_size = 5
step_size = 1

#create the files
if create_density:
    create_density_files(study, root, nimbal_drive, group_name[group], paper_path, master_subj_list,
                         window_size, step_size, create_stride_time)


if density_graph:  # density plot
    # path = nimbal_drive + demo_path
    # demodata = read_demo_ondri_data(path)
    # subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']

    # reorder based on meda step totals
    subj_total = subj_24hr.iloc[:, 0:2]  # selects subj and total median column
    subj_total = subj_total.iloc[1:].reset_index(drop=True)
    sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
    subj_list = sorted.iloc[:, 0]
    visit = '01'
    window_text = 'win_' + str(window_size) + 's_step_' + str(step_size) + 's_'

    # plot the bout density file
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    intensity_blocks = []
    for subj in subj_list:
        file = subj + '_' + visit + '_' + window_text + '_density.csv'
        density_subj = pd.read_csv(path_density + file)
        density_subj = density_subj.iloc[2:].reset_index(drop=True)
        density_subj = density_subj.iloc[:, 1:]
        rotated = density_subj.T
        intensity_blocks.append(rotated)
    intensity_matrix = pd.concat(intensity_blocks, axis=0, ignore_index=True)
    intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
    intensity_matrix = intensity_matrix.fillna(0)
    intensity_array = intensity_matrix.to_numpy(dtype=float)

    plt.figure(figsize=(10, 6))
    plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
    plt.colorbar(label='Intensity')
    plt.xlabel('Time (minutes) (midnight-midnight)')
    plt.ylabel('Subjects and days stacked')
    plt.title('Daily step density (all subjects/days)')
    plt.tight_layout()
    plt.show()
    print('pause')

if calc_pref_stride:
    # path = nimbal_drive + demo_path

    # reorder based on meda step totals
    subj_total = subj_24hr.iloc[:, 0:2]  # selects subj and total median column
    subj_total = subj_total.iloc[1:].reset_index(drop=True)
    sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
    subj_list = sorted.iloc[:, 0]
    visit = '01'
    window_text = 'win_' + str(window_size) + 's_step_' + str(step_size) + 's_'

    # calc and plot / file pre_density
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    intensity_blocks = []
    for subj in subj_list:
          file = subj + '_' + visit + '_' + window_text + '_density.csv'
          density_subj = pd.read_csv(path_density + file)
          density_subj = density_subj.iloc[2:].reset_index(drop=True)
          density_subj = density_subj.iloc[:, 1:]

          #convert to density  strides / sec
          density_subj = density_subj / window_size

          #calculate the mean/median/mode and STD for all data  > 0 but less than


          rotated = density_subj.T
          intensity_blocks.append(rotated)
            intensity_matrix = pd.concat(intensity_blocks, axis=0, ignore_index=True)
            intensity_matrix = intensity_matrix.apply(pd.to_numeric, errors='coerce')
            intensity_matrix = intensity_matrix.fillna(0)
            intensity_array = intensity_matrix.to_numpy(dtype=float)




          plt.figure(figsize=(10, 6))
          plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
            plt.colorbar(label='Intensity')
            plt.xlabel('Time (minutes) (midnight-midnight)')
            plt.ylabel('Subjects and days stacked')
            plt.title('Daily step density (all subjects/days)')
            plt.tight_layout()
            plt.show()
            print('pause')

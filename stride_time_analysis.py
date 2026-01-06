import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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

#which subjects?
master_subj_list = select_subjects(nimbal_drive, study)
#master_subj_list = ['OND09_SBH0006']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
#print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

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

# reorder based on meda step totals set subj list from steps files?
subj_total = subj_24hr.iloc[:, 0:2]  # selects subj and total median column
subj_total = subj_total.iloc[1:].reset_index(drop=True)
sorted = subj_total.sort_values(by=subj_total.columns[1]).reset_index(drop=True)
subj_list = sorted.iloc[:, 0]
print('Total # subjects for this analysis: \t' + str(len(subj_list)))

#values for density analysis
window_size = 10
step_size = 1
visit = '01'
window_text = 'win_' + str(window_size) + 's_step_' + str(step_size) + 's_'
print('\tWindow size (sec): '+ '\t' + str(window_size))
print('\tOverlap (sec): '+ '\t' + str(step_size) )

create_density = False
create_stride_time = False
calc_basic_stats = False
calc_preferred = False
density_graph = False
compare_density = True

#create the files
if create_density:
    create_density_files(study, root, nimbal_drive, group_name[group], paper_path, master_subj_list,
                         window_size, step_size, create_stride_time)

if density_graph:  # density plot
    # path = nimbal_drive + demo_path
    # demodata = read_demo_ondri_data(path)
    # subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']

    # plot the bout density file
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    intensity_blocks = []
    for subj in subj_list:
        file = subj + '_' + visit + '_' + window_text + '_density.csv'
        raw_data = pd.read_csv(path_density + file)
        density_subj = raw_data.iloc[2:].reset_index(drop=True)
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

if calc_preferred:
    max = 3.0
    min = 0.5

    #   path = nimbal_drive + demo_path

    # calc and plot / file pre_density
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    group_stats = []

    for subj in subj_list:
        print('\tSubject: \t'+subj)
        file = subj + '_' + visit + '_' + window_text + '_density.csv'
        raw_data = pd.read_csv(path_density + file)
        raw_data = raw_data.iloc[2:].reset_index(drop=True)
        raw_data = raw_data.iloc[:, 1:]

        #convert to density  strides / sec
        density_subj = raw_data / window_size

        # flatten to 1 array
        combined = density_subj.stack()
        #drop NA , 0 and values >
        #convert na to zeros

        cleaned = combined[(combined.notna()) & (combined != 0)]
        #cropped around min and max to narrow preferred calcualtion
        cropped = cleaned[(cleaned > min) & (cleaned < max)]

        stats = cropped.describe()
        # convert to a row and label it with subject ID
        stats_df = stats.to_frame().T
        stats_df["subject"] = subj
        group_stats.append(stats_df)

        #caluclate peak value from histogram need bin #s
        #counts, bins = np.histogram(cropped, bins=50)
        #peak_bin_index = counts.argmax()
        #peak_bin_start = bins[peak_bin_index]
        #peak_bin_end = bins[peak_bin_index + 1]
        #print("\tPeak bin:\t", peak_bin_start, "to", peak_bin_end)

        #calculate peak based on KDE
        #kde = gaussian_kde(cropped)
        # Evaluate KDE on a grid
        #x_grid = np.linspace(cropped.min(), cropped.max(), 1000)
        #density = kde(x_grid)

        # Peak = x where density is highest
        #peak_value = x_grid[np.argmax(density)]
        #print("\tKDE peak:\t", peak_value)
        #print ('done')


        '''
        plt.figure(figsize=(10, 6))
        plt.imshow(intensity_matrix, aspect='auto', cmap='viridis')  # or 'hot', 'plasma', 'magma'
        plt.colorbar(label='Intensity')
        plt.xlabel('Time (minutes) (midnight-midnight)')
        plt.ylabel('Subjects and days stacked')
        plt.title('Daily step density (all subjects/days)')
        plt.tight_layout()
        plt.show()
        print('pause')
        '''
    final_stats = pd.concat(group_stats, ignore_index=True)
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    file = 'All_' + visit + '_' + window_text + '_cropped_density_stats.csv'
    final_stats.to_csv(path_density+file, index=False)
    print (final_stats)

if compare_density:
    path_density = nimbal_drive + paper_path + 'Summary_data\\density\\' + study + '\\'
    file = 'All_' + visit + '_' + window_text + '_density_stats.csv'
    final_stats1 = pd.read_csv(path_density + 'All_01_win_10s_step_1s__density_stats.csv')
    final_stats2 = pd.read_csv(path_density + 'All_01_win_10s_step_1s__cropped_density_stats.csv')
    merged = final_stats1.merge(final_stats2, on='subject')
    corr = merged["50%_x"].corr(merged["50%_y"])
    print('n: '+str(len(merged))+ ' r: '+ str(corr))
    plt.scatter(merged["50%_x"], merged["50%_y"])
    plt.xlabel("df1 value")
    plt.ylabel("df2 value")
    plt.show()
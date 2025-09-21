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
#master_subj_list = ['OND09_SBH0006']
print('\nTotal # subjects: \t' + str(len(master_subj_list)) + '\n')
print('First 5 subject in list...' + str(master_subj_list[:5])+'\n')

#create summary data files
create = False
if create:
    time_window ='wake'#'1010' #'24hr' #'1010'  'wake'

    create_bin_density_files(time_window, study, root, nimbal_drive, paper_path, master_subj_list,
                             bin_list_steps, bin_width_time)

print ('pause')

calc_basic_stats = True
if calc_basic_stats:
    #select specific subjects from the study group
    #select ONDRI controls
    path = nimbal_drive + demo_path
    demodata = read_demo_ondri_data(path)
    subj_list = demodata[demodata['COHORT'] == 'Community Dwelling']['SUBJECT']
    group_name = 'Control'
    window = 'wake'

    step_vs_dur = False
    #calcualte medians and std for each bout and clustred bouts
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_list, group_name)

    step_vs_dur = True #run duraiton file
    calc_basic_stride_bouts_stats(step_vs_dur, nimbal_drive, study, window, paper_path, subj_list, group_name)


    #SML_median, SML_std, SML_pct_median, SML_pct_std = bouts_SML(nimbal_drive, study, window, paper_path, subj_list)

plot = True
figure1 = False #swarm totals
figure1b = False
figure2 = False  #KDE distibiton - bouts/unbouted
figure3 = False  #bout disitbution
figure4 = True

if plot:
    path = nimbal_drive + demo_path
    group_name = 'Control'
    study = 'OND09'

    path_24hr = nimbal_drive + paper_path + 'Summary_data\\' + study + '_24hr_' + group_name + '_bout_duration_'
    subj_24hr = pd.read_csv(path_24hr +'_subj_stats.csv', header=[0, 1])
    group_24hr = pd.read_csv(path_24hr +'_group_stats.csv')
    subj_pct_24hr = pd.read_csv(path_24hr + '_pct_subj_stats.csv', header=[0, 1])
    group_pct_24hr = pd.read_csv(path_24hr + '_pct_group_stats.csv')

    path_1010 = nimbal_drive + paper_path + 'Summary_data\\' + study + '_1010_' + group_name + '_bout_duration_'
    subj_1010 = pd.read_csv(path_1010 +'_subj_stats.csv', header=[0, 1] )
    group_1010 = pd.read_csv(path_1010 + '_group_stats.csv')
    subj_pct_1010 = pd.read_csv(path_1010 + '_pct_subj_stats.csv', header=[0, 1])
    group_pct_1010 = pd.read_csv(path_1010 + '_pct_group_stats.csv')

    path_wake = nimbal_drive + paper_path + 'Summary_data\\' + study + '_wake_' + group_name + '_bout_duration_'
    subj_wake = pd.read_csv(path_wake + '_subj_stats.csv', header=[0, 1])
    group_wake = pd.read_csv(path_wake + '_group_stats.csv')
    subj_pct_wake = pd.read_csv(path_wake + '_pct_subj_stats.csv', header=[0, 1])
    group_pct_wake = pd.read_csv(path_wake + '_pct_group_stats.csv')

    plot_24hr_all = subj_24hr[('window_total_strides','median')]
    plot_1010_all = subj_1010[('window_total_strides', 'median')]
    plot_wake_all = subj_wake[('window_total_strides', 'median')]

    plot_24hr_unbouted = subj_24hr[('window_not_bouted_strides', 'median')]
    plot_1010_unbouted = subj_1010[('window_not_bouted_strides', 'median')]
    plot_wake_unbouted = subj_wake[('window_not_bouted_strides', 'median')]

    plot_24hr_bouted = plot_24hr_all - plot_24hr_unbouted
    plot_1010_bouted = plot_1010_all - plot_1010_unbouted
    plot_wake_bouted = plot_wake_all - plot_wake_unbouted

    short_24hr_bouted = subj_24hr[[('strides_<_5', 'median'), ('strides_<_10', 'median'),('strides_<_30', 'median')]]
    med_24hr_bouted = subj_24hr[[('strides_<_60', 'median'), ('strides_<_180', 'median')]]
    long_24hr_bouted = subj_24hr[[('strides_<_600', 'median'), ('strides_>_600', 'median')]]

    night_time_totals = plot_24hr_all - plot_wake_all

    if figure1:
        # Sample DataFrame with two numeric columns
        df = pd.DataFrame({'24 HR': plot_24hr_all, 'Wake': plot_wake_all, '10AM-10PM': plot_1010_all})
        # Melt the DataFrame to long format for seaborn
        melted_df = df.melt(var_name='Window', value_name='Total steps')

        # Create the swarm plot
        plt.figure(figsize=(8, 6))
        #sns.swarmplot(x='Window', y='Total steps', data=melted_df, size=6)
        #sns.catplot(data= melted_df, x='Window', y='Total steps', kind='swarm',palette={'24 HR': 'skyblue', '10AM-10PM': 'salmon'})

        # Create violin plot
        sns.violinplot(x='Window', y='Total steps', data=melted_df, inner=None, palette={'24 HR': 'skyblue','Wake': 'magenta', '10AM-10PM': 'salmon'})
        # Overlay swarm plot
        sns.swarmplot(x='Window', y='Total steps', data=melted_df, color='black', size=4)

        plt.ylim(bottom=0)
        plt.title('Steps / day comparing time window')
        plt.xlabel('Window')
        plt.ylabel('Median unilateral steps / day')
        plt.tight_layout()
        plt.show()

    if figure1b:
        # Create the swarm plot
        plt.figure(figsize=(8, 6))
        # Create violin plot
        sns.violinplot(y=night_time_totals, inner=None, color='magenta')
        # Overlay swarm plot
        sns.swarmplot(y=night_time_totals, color='black', size=4)

        plt.ylim(bottom=0, top=1000)

        plt.title('Steps / night')
        plt.xlabel('Window')
        plt.ylabel('Median unilateral steps / night')
        plt.tight_layout()
        plt.show()

    if figure2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

        # Plot 1
        sns.kdeplot(x=plot_24hr_all, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[0])
        sns.kdeplot(x=plot_wake_all, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[0])
        sns.kdeplot(x=plot_1010_all, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[0])
        axes[0].set_title('Total unilateral steps')
        axes[0].set_xlabel('Step Count')
        axes[0].set_yticks([])
        axes[0].legend()

        # Plot 2
        sns.kdeplot(x=plot_24hr_unbouted, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[1])
        sns.kdeplot(x=plot_wake_unbouted, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[1])
        sns.kdeplot(x=plot_1010_unbouted, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[1])
        axes[1].set_title('Unbouted unilateral steps')
        axes[1].set_xlabel('Step Count')
        axes[1].set_yticks([])
        axes[1].legend()

        # Plot 3
        sns.kdeplot(x=plot_24hr_bouted, fill=True, label='24 HR', color='skyblue', alpha=0.5, ax=axes[2])
        sns.kdeplot(x=plot_wake_bouted, fill=True, label='Wake', color='magenta', alpha=0.5, ax=axes[2])
        sns.kdeplot(x=plot_1010_bouted, fill=True, label='10AM–10PM', color='salmon', alpha=0.5, ax=axes[2])
        axes[2].set_title('Bouted unilateral steps')
        axes[2].set_xlabel('Step Count')
        axes[2].set_yticks([])
        axes[2].legend()

        # Final layout
        fig.suptitle('Step Count Distributions Across Time Windows', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()

    if figure3:

        #plot_labels = ['Total', 'Unbouted', '<5', '5-10', '10-25', '25-50', '50-100', '100-300', '>300']
        plot_labels = ['Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']
        median_24hr_nototal = group_24hr.iloc[1:].reset_index(drop=True)
        median_24hr_pct_nototal = group_pct_24hr.iloc[1:].reset_index(drop=True)

        fig, axs = plt.subplots(2, figsize=(8, 9))
        # median std strides
        ticks = list(range(len(plot_labels)))
        axs[0].bar(median_24hr_nototal.index, median_24hr_nototal['Median'].values, yerr=median_24hr_nototal['Std'], capsize=5, color='lightblue', edgecolor='black')
        axs[0].set_title('Median unilateral steps / day by bout length')
        axs[0].set_xlabel('Bout length (sec)')
        axs[0].set_ylabel('Unilateral steps / day')
        axs[0].set_xticks(ticks=ticks, labels=plot_labels)
        axs[0].set_ylim(bottom=0)

        # median std strides
        ticks = list(range(len(plot_labels)))
        axs[1].bar(median_24hr_pct_nototal.index, median_24hr_pct_nototal['Median'].values, yerr=median_24hr_pct_nototal['Std'], capsize=5, color='lightgreen', edgecolor='black')
        axs[1].set_title('Percent median unilateral steps/day by bout length')
        axs[1].set_xlabel('Bout length (sec)')
        axs[1].set_ylabel('% of total unilateral steps / day')
        axs[1].set_xticks(ticks=ticks, labels=plot_labels)
        axs[1].set_ylim(bottom=0)

        plt.tight_layout()
        plt.show()

    if figure4:
        #plot_labels = ['Total','Unbouted', '<5', '5-10', '10-30', '30-60', '60-180', '180-600', '>600']

        short = short_24hr_bouted.sum(axis=1)
        med = med_24hr_bouted.sum(axis=1)
        long = long_24hr_bouted.sum(axis=1)

        fig, axs = plt.subplots(2,2, figsize=(8, 9), sharex=True, sharey=True)

        sns.scatterplot(x=plot_24hr_all, y=plot_24hr_unbouted, color='red', label='Unbouted', ax=axs[0,0])
        sns.scatterplot(x=plot_24hr_all, y=short, color='magenta', label='Short <30 sec', ax=axs[0,1])
        sns.scatterplot(x=plot_24hr_all, y=med, color='blue', label='Medium 30-180 sec', ax=axs[1,0])
        sns.scatterplot(x=plot_24hr_all, y=long, color='green', label='Long  > 180 sec',ax=axs[1,1])

        plt.title('Total steps versus bouted and unbouted steps')
        plt.xlabel('Total steps (median) / day')
        plt.ylabel('Median number steps / day')
        plt.tight_layout()
        plt.show()



    print('pause')
    '''
    #plot_stride_bouts_histogram(nstride_all_median, nstride_all_std, nstride_pct_all_median, nstride_pct_all_std,
    #                            totalTF=False)
    print ('pause')
    #bouts_SML(nimbal_drive, study, window, paper_path, subj_list)
    '''


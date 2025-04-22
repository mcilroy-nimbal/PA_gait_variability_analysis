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
import matplotlib.gridspec as gridspec
import datetime
import openpyxl

bland = True
gini = True
plot_gini_steps = True
plot_gini_groups = True

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

if gini:
    ###########################################
    #read in the cleaned data file for the HANNDS methods paper
    nimbal_dr = 'o:'
    new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
    #this woudl read in the elegible subejcts with demogrpahic data
    #demodata = read_orig_clean_demo()
    #Import data files - use this if file already created
    demodata = pd.read_csv(nimbal_dr+new_path+"OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv")

    #merge dual diagonis - other MCI
    demodata['COHORT'] = demodata['COHORT'].replace('MCI;CVD','CVD')
    demodata['COHORT'] = demodata['COHORT'].replace('MCI;PD','PD')
    demodata['COHORT'] = demodata['COHORT'].replace('AD;MCI','MCI')
    #collapse AD MCI
    demodata['COHORT'] = demodata['COHORT'].replace('AD','MCI')
    demodata['COHORT'] = demodata['COHORT'].replace('MCI','AD/MCI')

    ########################################################
    # loop through each eligible subject
    # File the time series in a paper specific forlder?
    #extract on a few variabels from the demo

    demodata = demodata[['SUBJECT','COHORT','AGE']]
    demodata['gini_steps', 'alpha_steps', 'xmin_steps','fit_steps', 'gini_dur', 'alpha_dur','xmin_dur', 'fit_dur'] = None
    all_steps=[]
    all_dur=[]
    for index, row in demodata.iterrows():
        print(f'\rFind subjs - Progress: {index}' + ' of ' + str(len(demodata)), end='', flush=True)
        #remove the underscoe that is in the subject code from the demodata file

        parts = row['SUBJECT'].split('_', 2)  # Split into at most 3 parts
        if len(parts) == 3:
            subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore

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
            all_steps.extend(bouts['step_count'])
            #g_steps, a_steps, xmin_steps, n_steps, fit_steps = alpha_gini_index (data, plot=False)
            #demodata.at[index,'gini_steps'] = g_steps
            #demodata.at[index, 'alpha_steps'] = a_steps
            #demodata.at[index, 'xmin_steps'] = xmin_steps
            #demodata.at[index, 'fit_steps'] = fit_steps
            #by duration
            bouts['start'] = pd.to_datetime(bouts['start_time'])
            bouts['end'] = pd.to_datetime(bouts['end_time'])

            # Calculate difference in seconds
            bouts['duration'] = (bouts['end'] - bouts['start']).dt.total_seconds()
            all_dur.extend(bouts['duration'].values.tolist())
            #g_dur, a_dur, xmin_dur, n_dur , fit_dur = alpha_gini_index(bouts['duration'], plot=False)
            #demodata.at[index, 'gini_dur'] = g_dur
            #demodata.at[index, 'alpha_dur'] = a_dur
            #demodata.at[index, 'xmin_dur'] = xmin_dur
            #demodata.at[index, 'fit_dur'] = fit_dur

    #demodata.to_csv(summary_path +'alpha_gini_bouts.csv')
    #all_df = pd.DataFrame({'steps': all_steps, 'dur': all_dur})

    #sns.histplot(x=all_steps, bins=100, kde=False)
    #drop strides <5
    #all_steps = [x for x in all_steps if x >= 4]
    #plt.hist(all_steps, bins=200)
    #plt.show()
    #all_steps = [x for x in all_steps if x >= 10]
    #sns.histplot(x=all_dur, bins=200, kde=False)
    #plt.show()


if plot_gini_steps:

    power_data = pd.read_csv(summary_path+'alpha_gini_bouts.csv')
    power_data['m_subj'] = power_data['SUBJECT'].str.replace('_', '')
    bout_data = pd.read_csv(summary_path + 'steps_daily_bins.csv')
    bout_data['m_subj'] = bout_data['subj'].str.replace('_', '')
    all_data = pd.merge(power_data, bout_data, on='m_subj', how='inner')
    print ('merged')
    #left_on='subject_id', right_on='participant_id',

    fig, axs = plt.subplots(3,3, figsize=(12,12))

    axs[0,0].hist(all_data['gini_steps'], bins=15, alpha=0.6, label='Step #', color='blue', edgecolor='black')
    axs[0,0].hist(all_data['gini_dur'], bins=15, alpha=0.6, label='Duration', color='orange', edgecolor='black')
    axs[0,0].set_title('Gini index (steps and duration)')
    axs[0,0].set_xlabel=('Gini Index')
    axs[0,0].set_ylabel = ('Frequency')
    axs[0,0].legend()

    axs[0,1].scatter(all_data['total'], all_data['gini_steps'], color='black', label='steps')
    axs[0,1].set_title('Gini versus total steps')
    axs[0,1].set_xlabel = ('Total steps')
    axs[0,1].set_ylabel = ('Gini')
    axs[0,1].legend()

    axs[0,2].scatter(all_data['<_3'], all_data['gini_steps'], color='red', label='steps')
    axs[0,2].set_title('Gini versus total steps')
    axs[0,2].set_xlabel = ('% steps <3')
    axs[0,2].set_ylabel = ('Gini')
    axs[0,2].legend()

    axs[1,0].hist(all_data['alpha_steps'], bins=15, alpha=0.6, label='Step #', color='blue', edgecolor='black')
    axs[1,0].hist(all_data['alpha_dur'], bins=15, alpha=0.6, label='Duration', color='orange', edgecolor='black')
    axs[1,0].set_title('Alpha (steps and duration)')
    axs[1,0].set_xlabel = ('Alpha')
    axs[1,0].set_ylabel = ('Frequency')
    axs[1,0].legend()

    axs[1, 1].scatter(all_data['total'], all_data['alpha_steps'], color='black', label='steps')
    axs[1, 1].set_title('Alpha versus total steps')
    axs[1, 1].set_xlabel = ('Total steps')
    axs[1, 1].set_ylabel = ('Alpha')
    axs[1, 1].legend()

    axs[1, 2].scatter(all_data['<_3'], all_data['alpha_steps'], color='red', label='steps')
    axs[1, 2].set_title('Gini versus total steps')
    axs[1, 2].set_xlabel = ('% steps <3')
    axs[1, 2].set_ylabel = ('Alpha')
    axs[1, 2].legend()


    axs[2,0].hist(all_data['xmin_steps'], bins=15, alpha=0.6, label='Step #', color='blue', edgecolor='black')
    axs[2,0].hist(all_data['xmin_dur'], bins=15, alpha=0.6, label='Duration', color='orange', edgecolor='black')
    axs[2,0].set_title('Xmin (steps and duration)')
    axs[2,0].set_xlabel = ('Xmin')
    axs[2,0].set_ylabel = ('Frequency')
    axs[2,0].legend()

    axs[2, 1].hist(all_data['fit_steps'], bins=15, alpha=0.6, label='Step #', color='blue', edgecolor='black')
    axs[2, 1].hist(all_data['fit_dur'], bins=15, alpha=0.6, label='Duration', color='orange', edgecolor='black')
    axs[2, 1].set_title('Fit (steps and duration)')
    axs[2, 1].set_xlabel = ('Fit')
    axs[2, 1].set_ylabel = ('Frequency')
    axs[2, 1].legend()

    axs[2,2].scatter(all_data['alpha_steps'], all_data['gini_steps'], color='black', label='steps')
    axs[2,2].scatter(all_data['alpha_dur'], all_data['gini_dur'], color='green', label='dur')
    axs[2,2].set_title('Alpha versus Gini')
    axs[2,2].set_xlabel = ('Alpha')
    axs[2,2].set_ylabel = ('Gini')
    axs[2,2].legend()


    plt.show()

if plot_gini_groups:
    power_data = pd.read_csv(summary_path + 'alpha_gini_bouts.csv')
    power_data['m_subj'] = power_data['SUBJECT'].str.replace('_', '')
    bout_data = pd.read_csv(summary_path + 'steps_daily_bins.csv')
    bout_data['m_subj'] = bout_data['subj'].str.replace('_', '')
    summary_bout = bout_data.groupby('subj', as_index=False).sum()
    summary_bout['m_subj'] = summary_bout['subj'].str.replace('_', '')
    all_data = pd.merge(power_data, summary_bout, on='m_subj', how='inner')
    # Define age bins and labels
    bins = [0, 35, 55, 65, 75, 85, 95]
    labels = [f"{bins[i]}–{bins[i + 1]}" for i in range(len(bins) - 1)]

    # Create new column for age group
    all_data['age_group'] = pd.cut(all_data['AGE'], bins=bins, labels=labels, right=False)
    print (all_data['age_group'].value_counts())

    sns.histplot(data=all_data, x='gini_steps', hue='COHORT', bins=20, kde=True, stat='density')

    plt.title("Histogram of Gini Index by Group")
    plt.xlabel("Gini Index (sec)")
    plt.ylabel("Density")
    #plt.legend(title="Group")
    plt.tight_layout()
    plt.show()

    sns.histplot(data=all_data, x='gini_steps', hue='age_group', bins=20, kde=True, stat='density')
    plt.title("Histogram of Gini Index by Group")
    plt.xlabel("Gini Index (sec)")
    plt.ylabel("Density")
    #plt.legend(title="Age")
    plt.tight_layout()
    plt.show()

    # Set up the figure
    #fig = plt.figure(figsize=(10, 6))
    #gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    # Main histogram
    #ax0 = plt.subplot(gs[0])
    #sns.histplot(data=all_data, x='gini_steps', hue='group', bins=20, multiple='dodge', ax=ax0)
    #ax0.set_xlabel('Gini_index')
    #ax0.set_ylabel('Density')

    # Side distribution
    #ax1 = plt.subplot(gs[1], sharey=ax0)
    #sns.kdeplot(data=all_data, y='gini_steps', hue='group', fill=True, ax=ax1, legend=False)
    #ax1.set_xlabel('Density')
    #ax1.set_ylabel('')
    #ax1.yaxis.tick_right()
    #ax1.yaxis.set_label_position("right")

    #plt.tight_layout()
    #plt.show()


if bland:
    #all_data['mean_gini'] = (all_data['gini_steps'] + all_data['gini_dur']) / 2
    all_data['diff_gini'] = all_data['gini_steps'] - all_data['gini_dur']

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=all_data, x='AGE', y='diff_gini', alpha=0.7)

    # Plot mean difference and limits of agreement
    mean_diff = all_data['diff_gini'].mean()
    std_diff = all_data['diff_gini'].std()

    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Diff Gini')
    plt.axhline(mean_diff + 1.96*std_diff, color='gray', linestyle='--', label='+1.96 SD')
    plt.axhline(mean_diff - 1.96*std_diff, color='gray', linestyle='--', label='-1.96 SD')

    plt.xlabel('Age')
    plt.ylabel('Difference (Steps − Duration)')
    plt.title('Bland-Altman Plot vs Age')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
''' this STEP3 runs the various variability analysis across eligible subjects and writes data to file

'''

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import wake_sleep, bout_bins, steps_by_day, step_density_sec,read_orig_fix_clean_demo, read_demo_ondri_data
from variability_analysis_functions import alpha_gini_index
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import datetime
import openpyxl

bland = False
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

#data_opt = pd.read_csv(summary_path + 'alpha_gini_bouts.csv')
#data_fixed = pd.read_csv(summary_path + 'alpha_gini_bouts_xmins.csv')
#gini_steps', 'alpha_steps', 'xmin_steps','fit_steps', 'gini_dur', 'alpha_dur','xmin_dur', 'fit_dur'


###########################################
#read in the cleaned data file for the HANNDS methods paper
nimbal_dr = 'o:'
new_path = '\\Papers_NEW_April9\\Shared_Common_data\\OND09\\'
demodata = read_demo_ondri_data(nimbal_dr, new_path)


#gini runs
bouts_not_density = False #if using bout data TRUE else False for density
bout_step = False # if using n steps in bout - False if using duration
set_xmin = -1 #-1 if no setting of XMIN

if bouts_not_density:
    source = 'bouts'
    if bout_step:
        type = 'nsteps'
    else:
        type = 'duration'
else:
    source = 'density'
    type = 'min'


if gini:

    ########################################################
    # loop through each eligible subject
    # File the time series in a paper specific forlder?
    #extract on a few variabels from the demo

    demodata = demodata[['SUBJECT','COHORT','AGE', 'EMPLOY_STATUS']]
    demodata['gini', 'alpha', 'xmin','fits', 'npts'] = None

    for index, row in demodata.iterrows():
        print(f'\rFind subjs - Progress: {index}' + ' of ' + str(len(demodata)), end='', flush=True)
        #remove the underscoe that is in the subject code from the demodata file

        parts = row['SUBJECT'].split('_', 2)  # Split into at most 3 parts
        if len(parts) == 3:
            subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
            visit = '01'

            if bouts_not_density:
                #get step data for subject
                try:
                    bouts = pd.read_csv(path1 + bout_path + subject + '_' + visit + '_GAIT_BOUTS.csv')
                    ''' study_code, subject_id, coll_id, gait_bout_num, start_time, end_time, step_count'''
                except:
                    #log_file.write('Steps file not found - Subject: '+subject+ '\n')
                    continue
                if bout_step:
                    #by step number
                    data = bouts['step_count']
                else:
                    #by duration
                    bouts['start'] = pd.to_datetime(bouts['start_time'])
                    bouts['end'] = pd.to_datetime(bouts['end_time'])
                    # Calculate difference in seconds
                    data = (bouts['end'] - bouts['start']).dt.total_seconds()

            else:
                #FIND ALL THE DENSITY FIELS THAT MACTH
                #read in density and append to one array
                #use that data for gini
                try:
                    #subejct has hyphen between OND)( and subj in this file name
                    density = pd.read_csv(summary_path+'density\\'+ subject + '_' + visit + '_1min_density.csv')

                except:
                    #log_file.write('Steps file not found - Subject: '+subject+ '\n')
                    continue
                #mergae all the data columns to one array and remove zeros
                data = density.to_numpy().flatten()
                #data = data[data != 0]

            #plt.hist(data, bins=30, alpha=0.5, density=True)
            sns.kdeplot(data, fill=False, bw_adjust=0.01)

            #if set_xmin == -1:
            #    gini_val, alpha_val, xmin_val, n_val, fit_val = alpha_gini_index (data, plot=False)
            #else:
            #    gini_val, alpha_val, xmin_val, n_val, fit_val = alpha_gini_index(data, plot=False, xmin=set_xmin)

            #demodata.at[index,'gini'] = gini_val
            #demodata.at[index, 'alpha'] = alpha_val
            #demodata.at[index, 'xmin'] = xmin_val
            #demodata.at[index, 'fit'] = fit_val
            #demodata.at[index, 'npts'] = n_val

    2
    plt.xlabel('Density steps/min -no zeros')
    plt.ylabel('Density')
    #plt.legend()
    #plt.grid(True)
    plt.show()
    #demodata.to_csv(summary_path +'alpha_gini_'+source+'_'+type+'.csv')


if plot_gini_steps:

    power_data = pd.read_csv(summary_path + 'alpha_gini_bouts_duration.csv')
    #power_data = pd.read_csv(summary_path+'alpha_gini_bouts_nsteps.csv')

    power_data = power_data.fillna('')
    power_data['m_subj'] = power_data['SUBJECT'].str.replace('_', '')
    bout_data = pd.read_csv(summary_path + 'steps_daily_bins.csv')
    bout_data['m_subj'] = bout_data['subj'].str.replace('_', '')
    bout_short = bout_data[['m_subj','total']]
    subject_sum = bout_short.groupby('m_subj').sum().reset_index()
    subject_n = bout_short.groupby('m_subj').size().reset_index()
    subject_n.columns.values[1] = 'n'
    bout_short = pd.merge(subject_n, subject_sum, on='m_subj')
    bout_short['total/day'] = bout_short['total']/bout_short['n']
    all_data = pd.merge(power_data, bout_short, on='m_subj', how='inner')

    med_gini = all_data['gini'].median()
    med_steps = all_data['total/day'].median()

    '''
    conditions = [(all_data['gini'] >= med_gini) & (all_data['total/day'] >= med_steps),
                (all_data['gini'] >= med_gini) & (all_data['total/day'] < med_steps),
                (all_data['gini'] < med_gini) & (all_data['total/day'] >= med_steps),
                (all_data['gini'] < med_gini) & (all_data['total/day'] < med_steps)]
    # Define category labels
    categories = ['Long bouts - Many steps','Long bouts - Few steps',
                  'Short bouts - Many steps','Short bouts - Few steps']
    '''
    conditions = [(all_data['AGE'] < 55),
                  (all_data['AGE'] >= 55) & (all_data['AGE'] < 65),
                  (all_data['AGE'] >= 65) & (all_data['AGE'] < 70),
                  (all_data['AGE'] >= 70) & (all_data['AGE'] < 75),
                  (all_data['AGE'] >= 75) & (all_data['AGE'] < 80),
                  (all_data['AGE'] >= 80) & (all_data['AGE'] < 90),
                  (all_data['AGE'] > 90)]

    # Define category labels
    categories = ['<55', '55-65', '65-70','70-75', '75-80','80-90', '>90']

    # Create new column
    all_data['accum_bin'] = np.select(conditions, categories, default='Unknown')

    sns.scatterplot(data=all_data, x='npts', y='gini', hue='COHORT', palette='Set2')
    #plt.axvline(x=med_steps, color='black', linestyle='--', linewidth=0.5)
    #plt.axhline(y=med_gini, color='black', linestyle='--', linewidth=0.5)

    plt.title("Number of bouts versus Gini Index")
    plt.xlabel("Steps")
    plt.ylabel("Gini")
    plt.legend(title='Category')
    plt.show()

    # Set axis limits for consistency across subplots
    xlim = (0, all_data['total/day'].max())
    ylim = (0,1)

    # Create subplots: one for each category
    categories = all_data['COHORT'].unique()
    fig, axes = plt.subplots(nrows=1, ncols=len(categories), figsize=(12, 4), sharex=True, sharey=True)

    # If only one category, axes will not be an array, so handle that case
    if len(categories) == 1:
        axes = [axes]

    # Loop through each category and create a scatter plot
    for ax, category in zip(axes, categories):
        # Filter data by category
        category_data = all_data[all_data['COHORT'] == category]

        # Plot scatter
        ax.scatter(category_data['total/day'], category_data['gini'], label=category, color='blue')

        # Add vertical and horizontal lines for each point in the category
        for _, row in category_data.iterrows():
            ax.axvline(x=med_steps, color='black', linestyle='--', linewidth=0.5)
            ax.axhline(y=med_gini, color='black', linestyle='--', linewidth=0.5)

        # Set title and axis labels
        ax.set_title(f"Category: {category}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Steps/day")
        ax.set_ylabel("Gini index")

        # Optional: add a legend
        ax.legend()

    # Final touch
    plt.suptitle("Scatter Plots by Category", y=1.05)
    plt.tight_layout()
    plt.show()






    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs[0, 1].scatter(all_data['total'], all_data['gini_steps'], color='black', label='steps')
    axs[0, 1].set_title('Gini versus total steps')
    axs[0, 1].set_xlabel = ('Total steps')
    axs[0, 1].set_ylabel = ('Gini')
    axs[0, 1].legend()









    fig, axs = plt.subplots(3,3, figsize=(12,12))

    axs[0,0].hist(all_data['gini_steps'], bins=15, alpha=0.6, label='Step #', color='blue', edgecolor='black')
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
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import (wake_sleep, bout_bins, steps_by_day, step_density_sec,read_orig_fix_clean_demo,
                       read_demo_ondri_data, summary_density_bins)
from variability_analysis_functions import alpha_gini_index
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
demodata = read_demo_ondri_data(nimbal_dr, new_path)

#study = 'OND09'
study = 'SA-PR01'

###########################################
# read 60sec density summary files
density_sum1 = pd.read_csv(summary_path+study+'_density_summary_v3.csv')
density_sum2 = pd.read_csv(summary_path+'density_summary_v3.csv')
full_density = pd.concat([density_sum1, density_sum2], ignore_index=True)

all_sum = full_density[full_density['day'] == 'all']
all_sum['vlow_corr'] = all_sum['vlow_total']/all_sum['n_total']
all_sum['low_corr'] = all_sum['low_total']/all_sum['n_total']
all_sum['med_corr'] = all_sum['med_total']/all_sum['n_total']
all_sum['high_corr'] = all_sum['high_total']/all_sum['n_total']
#all_sum = all_sum[all_sum['AGE'] > 75]


if False:
    cols = ['COHORT','vlow_corr','low_corr','med_corr','high_corr']
    labels = ['<5 ', '5-20 ', '20-40' , '>40 ']
    data = all_sum[cols]
    data['COHORT'] = data['COHORT'].replace({'CVD': 'NDD', 'AD/MCI': 'NDD','PD': 'NDD', 'ALS': 'NDD', 'FTD': 'NDD'})
    data = data[~data['COHORT'].isin(['none'])]
    #just plot the values averaged scross the different groups
    # Group by COHORT and calculate mean and std for each column
    cohorts = data['COHORT'].unique()
    x = np.arange(4)
    jitter_width = 0.05
    fig, ax = plt.subplots()
    cols = cols[1:]
    for index, cohort in enumerate(cohorts):
        sub = data[data['COHORT'] == cohort]
        means = sub[cols].mean()
        stds = sub[cols].std()
        x_jittered = x + index * jitter_width
        ax.errorbar(x_jittered, means, yerr=stds, label=f'{cohort}', fmt='-o', capsize=4)


    ax.set_title('Step density for COHORTS')
    ax.set_xlabel('Density (strides/min)')
    ax.set_ylabel('Proportion of wear time')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(title='COHORT')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

#################################################
# Clustering of data
################################################
if True:
    data = all_sum[all_sum['COHORT'].isin(['superager','control'])]
    cols = ['vlow_corr','low_corr','med_corr','high_corr']
    xlabels = ['<5 ', '5-20 ', '20-40', '>40 ']
    x = np.arange(4)
    data = data[cols]

#Normalize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

#Cluster with KMeans
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(scaled_data)
    data['cluster'] = kmeans.fit_predict(scaled_data)
    all_sum['cluster'] = data['cluster']

#Melt the DataFrame into long-form
    data_long = data.melt(id_vars='cluster', var_name='feature', value_name='value')

#Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_long, x='feature', y='value', hue='cluster', palette='Set2')

    plt.xlabel('Density (strides/min)')
    plt.ylabel('Proportion of wear time')
    plt.xticks(ticks=x, labels=xlabels)
    plt.title('Step density pattern clusters - ALL')
    plt.show()



if True:
    data = all_sum[all_sum['COHORT'].isin(['superager','control'])]

#Plot
    custom_palette = {'superager': 'skyblue', 'control': 'salmon'}
    plt.figure(figsize=(3, 5))
    sns.violinplot(x='COHORT', y = 'zero_total', data=data, palette=custom_palette )
    plt.title('Sample-entropy by Cohort')
    plt.xlabel('Cohort')
    plt.ylabel('Sample-entropy')
    plt.tight_layout()
    plt.show()
print ('done')

'''
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_long, x='feature', y='value', hue='cluster', palette='Set2')
    plt.title('Row-wise Feature Values Colored by Cluster')
    plt.show()

    unique_clusters = all_sum['cluster'].unique()
    for cluster_id in unique_clusters:
        subset = all_sum[all_sum['cluster'] == cluster_id]
        #mean, median and std - total strides
        mean_strides = subset['n_steps'].mean()
        median_strides = subset['n_steps'].median()
        std_strides = subset['n_steps'].std()
        #mean and std age
        mean_age = subset['AGE'].mean()
        std_age = subset['AGE'].std()
        #mean non-linear
        mean_age = subset['AGE'].mean()

        #% from each cohort
        counts = subset['COHORT'].value_counts()
        row = [{'Mean strides':mean_strides, 'median_strides': median_strides, 'Std-strides': std_strides,'Mean age': mean_age, 'Std age': std_age,
                'Mean Entropy':mean_entropy, 'Mean DFA': mean_dfa}]
        print (row)
        print (counts)
        #summary_df.loc[len(summary_df)] = row
        #summary_counts.append = counts
        print ('pause')
print ('done')
'''
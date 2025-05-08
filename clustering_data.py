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


###########################################
# read 60sec density summary files
density_sum = pd.read_csv(summary_path+'density_summary_v3.csv')

all_sum = density_sum[density_sum['day'] == 'all']

#data = all_sum[['vlow_total','low_total','med_total','high_total']]
all_sum['vlow_corr'] = all_sum['vlow_total']/all_sum['n_total']
all_sum['low_corr'] = all_sum['low_total']/all_sum['n_total']
all_sum['med_corr'] = all_sum['med_total']/all_sum['n_total']
all_sum['high_corr'] = all_sum['high_total']/all_sum['n_total']
data = all_sum[['vlow_corr','low_corr','med_corr','high_corr']]
#################################################
# Clustering of data
################################################
#Normalize
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#Cluster with KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(scaled_data)
data['cluster'] = kmeans.fit_predict(scaled_data)
all_sum['cluster'] = data['cluster']

#Melt the DataFrame into long-form
data_long = data.melt(id_vars='cluster', var_name='feature', value_name='value')

#Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_long, x='feature', y='value', hue='cluster', palette='Set2')
plt.title('Row-wise Feature Values Colored by Cluster')
plt.show()

#calculate group differences as a table

summary_df = all_sum.groupby('cluster').agg(count=('cluster', 'size'), Mean_stride=('n_steps', 'mean'),
                            Median_stride=('n_steps', 'median'), Std_stride=('n_steps', 'std'),
                            Mean_age=('AGE', 'mean'), Std_age=('AGE', 'std'),
                            Cohort=('COHORT', 'unique')).reset_index()


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
    #% from each cohort
    counts = subset['COHORT'].value_counts()
    row = [{'Mean strides':mean_strides, 'median_strides': median_strides, 'Std-strides': std_strides,'Mean age': mean_age, 'Std age': std_age}] + counts

print ('done')
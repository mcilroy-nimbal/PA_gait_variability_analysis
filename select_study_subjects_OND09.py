import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Functions import read_demo_ondri_data,corr_matrix_all_columns, clustering
from scipy.stats import gaussian_kde
import seaborn as sns
import datetime

###############################################################
study = 'OND09'
nimbal_dr = 'O:'

###########################################
#read in the cleaned data file for the HANNDS methods paper
demo_path = nimbal_dr+'\\Papers_NEW_April9\\Shared_Common_data\\'+study+'\\'
#reads this file - 'OND09_ALL_01_CLIN_DEMOG_2025_CLEAN_HANDDS_METHODS_N245.csv'
demodata = read_demo_ondri_data(demo_path)

def remove_underscore(s):
    parts = s.split('_')
    fixed = parts[0] +'_'+ parts[1] + parts[2]
    return fixed
demodata['SUBJECT'] = demodata['SUBJECT'].apply(remove_underscore)

###########################################
# read in step bout data
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
summary_path = nimbal_dr + paper_path + 'Summary_data\\'

summary_steps_file = study+'_bout_steps_daily_bins_with_unbouted.csv'
steps_data = pd.read_csv(summary_path+summary_steps_file)
steps_data = steps_data.rename(columns={'subj': 'SUBJECT'})

summary_dur_file = study +'_bout_width_daily_bins_with_unbouted.csv'
width_data = pd.read_csv(summary_path+summary_dur_file)
width_data = width_data.rename(columns={'subj': 'SUBJECT'})

##############################################
# select subset of main data based on
# Count the number of rows for each unique SubjectCode
by_step=False

if by_step:
    steps_all = steps_data[steps_data['all/sleep']=='all']
    graph_title1 = 'Bouts by stride count'
else:
    #using width in time
    steps_all = width_data[width_data['all/sleep']=='all']
    graph_title1 = 'Bouts by seconds'

counts = steps_all['SUBJECT'].value_counts().reset_index()
counts = counts.reset_index()

#table of the counts
merged = pd.merge(counts, demodata, on='SUBJECT')
merged = merged[merged['COHORT'] == 'Community Dwelling']

# Define age bins and labels
bins = [0, 50, 60, 70, 80, 100]
labels = ['<50', '50-60', '60-70', '70-80','>80']

# Create new column for age groups
merged['AgeGroup'] = pd.cut(merged['AGE'], bins=bins, labels=labels, right=False)

merged = merged[(merged['AGE'] > 49) & (merged['count'] > 6)]
print(len(merged))

# Summary calculations
sex_counts = merged['SEX'].value_counts(normalize=True).mul(100).round(1)
avg_age = merged['AGE'].mean()
std_age = merged['AGE'].std()
marital_counts = merged['MRTL_STATUS'].value_counts(normalize=True).mul(100).round(1)

# Get unique values in 'Category'
a = merged['MRTL_STATUS'].unique()
mrtl_list = a.tolist()

work_counts = merged['EMPLOY_STATUS'].value_counts(normalize=True).mul(100).round(1)
a = merged['EMPLOY_STATUS'].unique()
employ_list = a.tolist()

# Build summary table
summary = pd.DataFrame({
    'Metric': ['N','Avg age', 'Std_age', '% Male', '% Female'] + [f'Count in {label}' for label in mrtl_list]+ [f'Count in {label2}' for label2 in employ_list],
    'Value': [
        len(merged),
        round(avg_age, 2),
        round(std_age, 2),
        sex_counts.get('Male', 0),
        sex_counts.get('Female', 0),
        *[marital_counts.get(label, 0) for label in mrtl_list],
        *[work_counts.get(label, 0) for label in employ_list]
    ]
})

# Create a blank row
blank_row = pd.DataFrame({'Metric': [''], 'Value': ['']})
summary = pd.concat([summary.iloc[:4], blank_row, summary.iloc[4:]]).reset_index(drop=True)
summary = pd.concat([summary.iloc[:11], blank_row, summary.iloc[11:]]).reset_index(drop=True)
print (summary)

target_subj = merged['SUBJECT'].unique()
subjects = pd.DataFrame(target_subj)
subjects.columns = ['SUBJECT']

#write the subject #s for the paper to file
subjects.to_csv(summary_path+'subject_ids_for_paper.csv')

#write the subject #s for the paper to file
merged.to_csv(summary_path+'subject_demo_for_paper.csv')

#write the cohort to file
summary.to_csv(summary_path+'subject_demo_summary_table.csv')

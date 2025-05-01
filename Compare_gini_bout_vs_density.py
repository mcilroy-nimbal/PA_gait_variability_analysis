
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
gini = False
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


bout_gini_steps = pd.read_csv(summary_path + 'alpha_gini_bouts_nsteps.csv')
bout_gini_dur = pd.read_csv(summary_path + 'alpha_gini_bouts_duration.csv')
bout_gini_density = pd.read_csv(summary_path + 'alpha_gini_density_min.csv')

#'gini', 'alpha', 'xmin', 'fit', 'npts'

merged = pd.merge(bout_gini_density, bout_gini_steps, on='SUBJECT', how='inner')

# Plot reaction_time vs accuracy
plt.scatter(merged['gini_x'], merged['gini_y'])
plt.xlabel('Gini - density')
plt.ylabel('Gini - bout steps')
plt.show()

plt.scatter(merged['gini_x'], merged['gini_y'])
plt.xlabel('Gini - density')
plt.ylabel('Gini - bout steps')
plt.show()

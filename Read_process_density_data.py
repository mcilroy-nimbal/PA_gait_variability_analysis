''' this
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
from scipy.stats import gaussian_kde
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



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

#gini runs
bouts_not_density = False #if using bout data TRUE else False for density
bout_step = False # if using n steps in bout - False if using duration
set_xmin = -1 #-1 if no setting of XMIN

source = 'density'
type = 'min'
dur_type ='1min'
#dur_type ='15sec'

########################################################
# loop through each eligible subject
# File the time series in a paper specific forlder?
#extract on a few variabels from the demo
demodata = demodata[['SUBJECT','COHORT','AGE', 'EMPLOY_STATUS']]
demodata['gini', 'alpha', 'xmin','fits', 'npts'] = None

#with PdfPages(summary_path+'\\all_total_1min.pdf') as pdf:

kde_x = np.linspace(0, 80, 100)
kdes = []
ages = []
cohorts = []

for index, row in demodata.iterrows():
    print(f'\rFind subjs - Progress: {index}' + ' of ' + str(len(demodata)), end='', flush=True)
    #remove the underscoe that is in the subject code from the demodata file


    parts = row['SUBJECT'].split('_', 2)  # Split into at most 3 parts
    if len(parts) == 3:
        subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
        visit = '01'

        #FIND ALL THE DENSITY FIELS THAT MACTH
        #read in density and append to one array
        #use that data for gini
        try:
            #subejct has hyphen between OND)( and subj in this file name
            density = pd.read_csv(summary_path+'density\\'+ subject + '_' + visit + '_'+ dur_type+'_density.csv')

        except:
            #log_file.write('Steps file not found - Subject: '+subject+ '\n')
            continue

        #loop through
        density = density.iloc[:,1:]
        density = density[density != 0]

        #fig = plt.figure(figsize=(12, 6))

        #for i, col in enumerate(density.columns):

        #    signal = density[col].values
        #    signal = signal[~np.isnan(signal)]
        #    kde = gaussian_kde(signal, bw_method='scott')
        #    kdes.append()
        #    plt.plot(kde_x, kde(kde_x))

        #mergae all the data columns to one array and remove zeros
        data = density.to_numpy().flatten()
        data = data[~np.isnan(data)]
        kde = gaussian_kde(data, bw_method='scott')
        kde_values = kde(kde_x)
        kdes.append(kde_values)
        ages.append(row['AGE'])
        cohorts.append(row['COHORT'])
        #plt.plot(kde_x, kde(kde_x), color='black', linewidth='3')
        #plt.title('Subj: '+str(row['SUBJECT']) +'  Cohort: '+str(row['COHORT'])+' Age: '+str(row['AGE']))
        #plt.ylim(0, 0.2)
        #pdf.savefig()  # Save to PDF
        #plt.close()
        #plt.show()

cohorts = np.array(cohorts)
ages = np.array(ages)
kdes = np.array(kdes)  # Shape: (n_samples, len(x_grid))

# Step 3: Normalize
scaler = StandardScaler()
kdes_scaled = scaler.fit_transform(kdes)

# Step 4: Cluster with KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(kdes_scaled)

# Step 4: Plot subplots
unique_labels = np.unique(labels)
n_clusters = len(unique_labels)

fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 4 * n_clusters), sharex=True)

if n_clusters == 1:
    axes = [axes]  # Ensure axes is iterable

for idx, cluster_id in enumerate(unique_labels):
    ax = axes[idx]
    indices = np.where(labels == cluster_id)[0]

    nsubj = len(indices)
    age = ages[indices].mean()
    cohort = cohorts[indices]
    unique_vals, counts = np.unique(cohort, return_counts=True)
    plt_head = 'n: '+str(nsubj)+' Age: '+ str(round(age,1))
    for val, count in zip(unique_vals, counts):
        plt_head = plt_head + '  Grp: '+ val + '% '+str(round(100 * count / nsubj,1))

    for i in indices:
        ax.plot(kde_x, kdes[i], alpha=0.7)

    ax.set_title(plt_head)
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("x")  # Label bottom subplot x-axis
plt.tight_layout()
plt.show()

print ('pause')

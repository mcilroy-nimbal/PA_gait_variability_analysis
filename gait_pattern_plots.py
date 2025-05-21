import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime

###############################################################
#
study = 'OND09'
#study = 'SA-PR01'

#set up paths
root = 'W:'

#check - but use this one - \prd\nimbalwear\OND09
if study == 'OND09':
    path1 = root+'\\prd\\NiMBaLWEAR\\OND09\\analytics\\'
else:
    path1 = root+'\\prd\\NiMBaLWEAR021\\SA-PR01\\analytics\\'

nimbal_drive = 'O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
summary_path = nimbal_drive + paper_path + 'Summary_data\\'
summary_steps_file = 'OND09_bout_steps_daily_bins_with_unbouted.csv'
summary_dur_file = 'OND09_bout_width_daily_bins_with_unbouted.csv'

#set the bin widths fro step/strides counting
bin_list_steps = [3, 5, 10, 20, 50, 100, 300]
bin_width_time = [5, 10, 15, 30, 60, 180, 600]


#create header
bin=[]
for k in range(len(bin_list_steps)):
    new = f'{'<'}_{bin_list_steps[k]}'
    bin.append(new)
last = '>_' + str(bin_list_steps[len(bin_list_steps)-1])
bin.append(last)
step_n_header = ['n_' + item for item in bin]
step_tot_header = ['strides_' + item for item in bin]

bin=[]
for k in range(len(bin_width_time)):
    new = f'{'<'}_{bin_width_time[k]}'
    bin.append(new)
last = '>_' + str(bin_width_time[len(bin_width_time)-1])
bin.append(last)
dur_n_header = ['n_' + item for item in bin]
dur_tot_header = ['strides_' + item for item in bin]

vars_step_tot = ['subj','not_bouted'] + step_tot_header
vars_dur_tot = ['subj','not_bouted'] + dur_tot_header

#var_list1_step_n = step_n_header

#######################################################
# mean values (across days - all values from bouts)
steps = True

if steps:
    file = summary_steps_file
else:
    file = summary_dur_file

bout_data = pd.read_csv(summary_path+file)

all = bout_data[bout_data['all/sleep'] == 'all']

if steps:
     subset = all[vars_step_tot]
else:
    subset = all[vars_dur_tot]

grouped = subset.groupby('subj').agg(['mean', 'std'])

means = grouped.xs('mean', axis=1, level=1)
stds = grouped.xs('std', axis=1, level=1)
cvs = stds / means

group_means = means.mean()
group_std = means.std()
group_cvs = cvs.mean()
group_cvs_sd = cvs.std()

# Plot
plt.figure(figsize=(8, 6))

data = group_means
data_std = group_std

plt.bar(data.index, data.values, color='blue')
plt.errorbar(data.index,data.values, yerr=[np.zeros_like(data_std.values), data_std.values],
            fmt='none', ecolor='black', capsize=5 )
for i, col in enumerate(means.columns):
    x = np.random.normal(i, 0.05, size=len(means[col]))  # jittered x-values
    plt.scatter(x, means[col], color='orange', alpha=0.7, label='Data Points' if i == 0 else "")


plt.ylabel('Coefficient of variation')
plt.title('Average Coefficient of variation (across days) per bout')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
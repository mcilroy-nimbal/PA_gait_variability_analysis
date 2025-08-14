import pandas as pd
from Functions import clustering, get_demo_characteristics, create_table
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import matplotlib.gridspec as gridspec


###############################################################
study = 'SA-PR01'
nimbal_dr = 'O:'
sub_study = 'AAIC 2025'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'
summary_path = nimbal_dr + paper_path + 'Summary_data\\'


##############################################################
#read in subject list from file in summary drive
subjects = pd.read_csv(summary_path+'subject_ids_'+sub_study+'.csv')

##############################################################
#get demogrpahci data
print ('reading demo data....')
demo_data = get_demo_characteristics(study, sub_study)
# Summary calculations
categ_vars = ['GROUP', 'sex', 'race', 'mc_employment_status', 'maristat', 'livsitua', 'independ','currently_exercise', 'currently_exercise_specify']
cont_vars = ['age_at_visit', 'educ', 'lsq_total', 'global_psqi', 'adlq_totalscore']
demo_data[cont_vars] = demo_data[cont_vars].apply(pd.to_numeric, errors='coerce')



#################################################################
#Figure 2 of poster

daily = pd.read_csv('W:\\SuperAging\\data\\summary\\AAIC 2025\\conference\\SA-PR01_daily_summary.csv')

list = ['subject_id', 'ankle_wear_duration', 'wrist_wear_duration', 'total_steps','moderate','vigorous','sleep_duration_total']
daily = daily[list]
daily = daily.rename(columns={'subject_id': 'SUBJECT'})

#Drop days that are too short
threshold = 72000
daily['total_steps'] = daily['total_steps'].where(daily['ankle_wear_duration'] >= threshold)
daily['moderate'] = daily['moderate'].where(daily['wrist_wear_duration'] >= threshold)
daily['moderate'] = daily['moderate']/60
daily['sleep_duration_total'] = daily['sleep_duration_total']/3600
summary = daily.groupby('SUBJECT').agg(['median', 'std']).reset_index()
summary = summary[summary['SUBJECT'].isin(subjects['SUBJECT'])]

summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.rename(columns={'SUBJECT_': 'SUBJECT'})
summary = summary.merge(demo_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left')


#fig, ax = plt.subplots(figsize=(12, 4))
#xcol = 'sleep_duration_total_median'
#ycol = 'sleep_duration_total_std'
#xcol = 'moderate_median'
#ycol = 'moderate_std'
ycol = 'total_steps_std'
xcol = 'total_steps_median'

huecol = 'GROUP'
#sns.jointplot(data=summary,x=xcol,y=ycol, hue='GROUP', marginal_kws=dict(common_norm=False, fill=True))

#ax.set_ylabel('Median Interday CofV (%)')
#ax.set_ylabel('Median steps / day', fontsize=24)
#ax.set_xlabel('Inter-day variation - steps (std)', fontsize=24)

#ax.set_title('Median Interday CofV vs bout lengths')
#ax.set_title('Inter-day CofV vs bout lengths', fontsize=32)
#ax.legend(title='Group',title_fontsize=20, fontsize=18)


fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

# Axes for plots
ax_main = fig.add_subplot(gs[1, 0])  # main scatterplot
ax_xdist = fig.add_subplot(gs[0, 0], sharex=ax_main)  # top histogram
ax_ydist = fig.add_subplot(gs[1, 1], sharey=ax_main)  # right histogram

# Plot scatter and regression lines per group
for group in summary[huecol].unique():
    group_df = summary[summary[huecol] == group]
    # Scatter
    sns.scatterplot(data=group_df, x=xcol, y=ycol, ax=ax_main, label=group)
    # Regression
    sns.regplot(data=group_df, x=xcol, y=ycol, ax=ax_main, scatter=False)

# Plot marginal distributions
sns.histplot(data=summary, x=xcol, hue=huecol, ax=ax_xdist, element="step", common_norm=False, legend=False)
sns.histplot(data=summary, y=ycol, hue=huecol, ax=ax_ydist, element="step", common_norm=False, legend=False)

# Clean up aesthetics
ax_xdist.axis("off")
ax_ydist.axis("off")
#ax_main.set_xticks(fontsize=24)
#ax_main.set_yticks(fontsize=24)
#ax_main.set_xlabel('Median sleep duration (hours / day)', fontsize=20)
#ax_main.set_ylabel('Inter-day sleep variability (std)', fontsize=20)
ax_main.legend(title=huecol)

#plt.suptitle("Grouped Regression with Marginal Distributions", y=0.95)
plt.show()
print ('pause')
'''
'''
###########################################
# read in step bout data
print ('reading bout data....')
summary_steps_file = study+'_bout_steps_daily_bins_with_unbouted.csv'
steps_data = pd.read_csv(summary_path+summary_steps_file)

summary_dur_file = study +'_bout_width_daily_bins_with_unbouted.csv'
width_data = pd.read_csv(summary_path+summary_dur_file)
width_data = width_data.rename(columns={'subj': 'SUBJECT'})

##############################################
# Run the analysis on bout steps or bout width
print ('Summarizing bouts data by subject (collapsed across days)...')
by_step = False
if by_step:
    steps_all = steps_data[steps_data['all/sleep']=='all']
    graph_title1 = 'Bouts by stride count'

else:
    #using width in time
    steps_all = width_data[width_data['all/sleep']=='all']
    graph_title1 = 'Bouts by seconds'


counts = steps_all['SUBJECT'].value_counts().reset_index()
counts = counts.reset_index()


################################################################################
# this sets upf fiels for plott ing and analysis
#select all steps that match the subject list
step1 = steps_all[steps_all['SUBJECT'].isin(subjects['SUBJECT'])]
# select only those with first 7 days
#step1 = step1.groupby('SUBJECT').head(7).reset_index(drop=True)

#slect the columns names to plot/analyze - only stride counts fo this
#this lien was for numebr of bouts
#select1 = step1.columns[step1.columns.str.startswith('n_')].tolist()

#these are the strides per bout class
select2 = step1.columns[step1.columns.str.startswith('strides_')].tolist()
#this adds unbouted
select2.insert(0, 'not_bouted')
select2.insert(0, 'total')

#calcualte the percentrage of steps in bouts relatiev to total (daily)
select3 = []
for col in select2:
    step1[col + '_pct'] = step1[col] / step1['total'] * 100
    select3.append(col+'_pct')



#######################################################################
#colaspe for cluster analysis

#collapse data into low, med, high bin widht durations
short_bouts = step1[['strides_<_5', 'strides_<_10', 'strides_<_30']]
step1['short'] = short_bouts.sum(axis=1)
med_bouts = step1[['strides_<_60', 'strides_<_180']]
step1['medium'] = med_bouts.sum(axis=1)
long_bouts = step1[['strides_<_600', 'strides_>_600']]
step1['long'] = long_bouts.sum(axis=1)


#####################################################################
#CLUSTER analysis
#select subset for persposes of clustering
cluster_cols = ['not_bouted', 'short', 'medium', 'long']
all_subset = step1[['SUBJECT'] + cluster_cols]

#subset = subset.groupby(subset['SUBJECT']).mean()

subset = all_subset.groupby(all_subset['SUBJECT']).agg(['sum','count'])
subset.columns = ['_'.join(col) for col in subset.columns]

subset['total'] = subset[['not_bouted_sum', 'short_sum', 'medium_sum', 'long_sum']].sum(axis=1)
subset['median_daily'] = subset['total'] / subset['not_bouted_count']
corr = subset['median_daily']
subset['not_bouted'] = corr * subset['not_bouted_sum'] /subset['total']
subset['short'] = corr * subset['short_sum'] /subset['total']
subset['medium'] = corr * subset['medium_sum'] /subset['total']
subset['long'] = corr * subset['long_sum'] /subset['total']

subset_cluster = subset[['not_bouted', 'short','medium','long']]

#sum across subject days
print ('Run and plot cluster analysis....')
cluster_data = subset_cluster
ncluster = 3
data_out, labels = clustering(cluster_data, ncluster=ncluster)
subject_clusters = pd.DataFrame({'SUBJECT': cluster_data.index,'GROUP': labels})
subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({0: 'Low'})
subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({1: 'High'})
subject_clusters['GROUP'] = subject_clusters['GROUP'].replace({2: 'High/Low'})
data_out['cluster'] = data_out['cluster'].replace({0: 'Low'})
data_out['cluster'] = data_out['cluster'].replace({1: 'High'})
data_out['cluster'] = data_out['cluster'].replace({2: 'High/Low'})

plot_cluster = False
if plot_cluster:
    x = np.arange(len(cluster_cols))
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data_out, x='feature', y='value', hue='cluster', hue_order=['Low','High/Low','High'], palette='Set2', linewidth=5)
    plt.xlabel('Bout durations', fontsize=24)
    plt.ylabel('Strides/day', fontsize=24)
    plt.xticks(ticks=x, labels=cluster_cols, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title('Bout pattern clusters', fontsize=32)
    plt.legend(title='Cluster', title_fontsize=20, fontsize=18)
    plt.show()


########################################################################
#determien charcateristic of group members 

#features indivdiuasl in the clusters - table 2
#add the cluster column to demo data
subject_clusters = subject_clusters.rename(columns={'GROUP': 'CLUSTER'})
demo_data = demo_data.merge(
    subject_clusters[['SUBJECT', 'CLUSTER']],  # Only bring in needed column
    on='SUBJECT',
    how='left')  # Keep all rows in demo_data

demo_data['maristat'] = demo_data['maristat'].apply(lambda x: 1 if x == 'Married' or x == 'Living as married/domestic partner' else 2)
demo_data['race'] = demo_data['race'].apply(lambda x: 1 if x == '1' else 2)
demo_data['sex'] = demo_data['sex'].apply(lambda x: 1 if x == 'Female' else 2)
demo_data['livsitua'] = demo_data['livsitua'].apply(lambda x: 1 if x == 'Lives alone' else 2)
demo_data['currently_exercise_specify'] = demo_data['currently_exercise_specify'].apply(lambda x: 1 if
        x == 'Every day' or x == 'At least 3x / week' else 2)

cont_vars = ['age_at_visit', 'educ', 'lsq_total', 'global_psqi', 'adlq_totalscore']

group_counts = demo_data.groupby('CLUSTER')['GROUP'].value_counts().unstack()
print(group_counts)
sex_counts = demo_data.groupby('CLUSTER')['sex'].value_counts().unstack()
print(sex_counts)
race_counts = demo_data.groupby('CLUSTER')['race'].value_counts().unstack()
print(race_counts)
marital_counts = demo_data.groupby('CLUSTER')['maristat'].value_counts().unstack()
print(marital_counts)

living = demo_data.groupby('CLUSTER')['livsitua'].value_counts().unstack()
print(living)
exercise = demo_data.groupby('CLUSTER')['currently_exercise_specify'].value_counts().unstack()
print(exercise)


age_stats = demo_data.groupby('CLUSTER')['age_at_visit'].agg(['mean', 'std']).round(4)
print(age_stats)
edu_stats = demo_data.groupby('CLUSTER')['educ'].agg(['mean', 'std']).round(4)
print(edu_stats)
adl_stats = demo_data.groupby('CLUSTER')['adlq_totalscore'].agg(['mean', 'std']).round(4)
print(adl_stats)
adl_stats = demo_data.groupby('CLUSTER')['lsq_total'].agg(['mean', 'std']).round(4)
print(adl_stats)


subset = subset.merge(
    subject_clusters[['SUBJECT', 'CLUSTER']],  # Only bring in needed column
    on='SUBJECT',
    how='left')  # Keep all rows in demo_data
subset['median_daily'] = subset['median_daily'] * 2
daily_stats = subset.groupby('CLUSTER')['median_daily'].agg(['mean', 'std','max', 'min']).round(4)
print(daily_stats)


#Table 1
cont_vars = ['AGE', 'YEARS_EDU', 'ADL_SCORE']

categ_vars_table1 = ['sex', 'race', 'maristat']
cont_vars_table1 = ['age_at_visit', 'educ', 'lsq_total', 'adlq_totalscore']
demo_data['maristat'] = demo_data['maristat'].apply(lambda x: 1 if x == 'Married' or x == 'Living as married/domestic partner' else 2)
demo_data['race'] = demo_data['race'].apply(lambda x: 1 if x == '1' else 2)
demo_data['sex'] = demo_data['sex'].apply(lambda x: 1 if x == 'Female' else 2)

group1 = demo_data[demo_data['GROUP'] == 'control']
group2 = demo_data[demo_data['GROUP'] == 'superager']

results = []
for var in cont_vars_table1:
    stat, pval = ttest_ind(group1[var], group2[var], nan_policy='omit')
    mean1 = group1[var].mean()
    mean2 = group2[var].mean()
    std1 = group1[var].std()
    std2 = group2[var].std()
    results.append({
        'Variable': var,
        'Mean_Group1': mean1,
        'SD_Group1': std1,
        'Mean_Group2': mean2,
        'SD_Group2': std2,
        'P-Value': pval
    })

cont_summary = pd.DataFrame(results)
cat_results = []

for var in categ_vars_table1:

    contingency = pd.crosstab(demo_data[var], demo_data['GROUP'])
    chi2, pval, _, _ = chi2_contingency(contingency)

    prct1 = (group1[var] == 1).mean() * 100
    prct2 = (group2[var] == 1).mean() * 100

    cat_results.append({
        'Variable': var,
        'Group1_%': prct1,
        'Group2_%': prct2,
        'P-Value': pval
    })

cat_summary = pd.DataFrame(cat_results)
final_summary = pd.concat([cont_summary, cat_summary], ignore_index=True)

create_table=False
if create_table:
    categ_table = pd.DataFrame()
    cont_table = pd.DataFrame()
    for i in range(ncluster):
        cluster_id = subject_clusters[subject_clusters['GROUP'] == i]
        #subjects = cluster_id['Subject']
        print ('Cluster '+str(i) + ' - n: '+ str(len(subjects)))

        cluster_demo = demo_data[demo_data['SUBJECT'].isin(cluster_id['Subject'])]
        categ, cont = create_table(cluster_demo, cont_vars, categ_vars)
        categ_table = pd.concat([categ_table, categ], ignore_index=True)
        cont_table = pd.concat([cont_table, cont], ignore_index=True)
        print ('pause')


    #table 1
    #variables - age, education, sex, race,
    categ_table.to_csv(summary_path+'cluster_demo_categ_updated.csv', index=False)
    cont_table.to_csv(summary_path+'cluster_demo_cont_updated.csv', index=False)
print ('test')
'''


#####################################################################
#means graph of all bout widths - uses medians for each subject and then medianas and STD
#MEANS GRAPH

#bouts setp #s absolute
select = select2
subj_sum = step1.groupby('SUBJECT')[select].sum()
subj_median = step1.groupby('SUBJECT')[select].median()
subj_mean = step1.groupby('SUBJECT')[select].mean()
subj_std = step1.groupby('SUBJECT')[select].std()
subj_CV = 100 * subj_std/subj_mean


##########################################################################
#plot data
group_data = demo_data
#group_data = subject_clusters
#if grouping by SA/CONTROl
subj_median = subj_median.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')
subj_mean = subj_mean.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')
subj_std = subj_std.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')
subj_CV = subj_CV.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')

#change this line for mean vs median
group_mean = subj_mean.groupby('GROUP').mean()

#chaneg this lien for man versus median
group_std = subj_mean.groupby('GROUP').std()

group_CV = subj_CV.groupby('GROUP').mean()
group_CV_std = subj_CV.groupby('GROUP').std()

#plotting
# Get list of variables to plot
group_mean = group_mean.drop(columns='total')
group_std = group_std.drop(columns='total')
group_CV = group_CV.drop(columns='total')
group_CV_std = group_CV_std.drop(columns='total')

variables = group_mean.columns
groups = group_mean.index
n_groups = len(groups)
n_vars = len(variables)

x = np.arange(n_vars)  # x-axis: one spot per variable
width = 0.8 / n_groups  # total bar width shared among groups

fig, ax = plt.subplots(figsize=(12, 4))

# Plot bars for each group
for i, group in enumerate(groups):
    # Heights and errors for this group
    heights = group_CV.loc[group].values
    #heights = group_CV.loc[group].values

    errors = group_CV_std.loc[group].values
    yerr = [np.zeros_like(errors), errors]
    #errors = group_CV_std.loc[group].values

    # X locations: offset for each group
    x_pos = x + (i - n_groups / 2) * width + width / 2

    ax.bar(x_pos,heights,yerr=yerr,capsize=5,width=width,label=f'{group}')

# Axis labels and ticks
ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7],
              labels=['Unbouted', '<5 s', '5-10 s', '10-30 s', '30-60 s', '60-180 s', '180-600 s', '>600 s'],
              fontsize=18)

#ax.set_ylabel('Median Interday CofV (%)', fontsize=24)
ax.set_ylabel('Inter-day CofV', fontsize=24)

ax.set_xlabel('Bout duration', fontsize=24)

#ax.set_title('Median Interday CofV vs bout lengths')
ax.set_title('Inter-day CofV vs bout lengths', fontsize=32)
ax.legend(title='Group',title_fontsize=20, fontsize=18)
plt.tight_layout()
plt.show()


by_group = True
if by_group:
    #group_data = demo_data
    group_data = subject_clusters

    #if grouping by SA/CONTROl
    subj_medians = subj_medians.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')

    subj_std = subj_std.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')
    subj_means = subj_means.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')
    subj_CV = subj_CV.merge(group_data[['SUBJECT', 'GROUP']], on='SUBJECT', how='left').drop(columns='SUBJECT')

    group_medians = subj_medians.groupby('GROUP').median()
    group_std = subj_std.groupby('GROUP').median()
    group_CV = subj_CV.groupby('GROUP').median()
    group_CV_std = subj_CV.groupby('GROUP').std()


    #plotting
    # Get list of variables to plot
    variables = group_medians.columns
    groups = group_medians.index
    n_groups = len(groups)
    n_vars = len(variables)

    x = np.arange(n_vars)  # x-axis: one spot per variable
    width = 0.8 / n_groups  # total bar width shared among groups

    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot bars for each group
    for i, group in enumerate(groups):
        # Heights and errors for this group
        heights = group_medians.loc[group].values
        #heights = group_CV.loc[group].values

        errors = group_std.loc[group].values
        #errors = group_CV_std.loc[group].values

        # X locations: offset for each group
        x_pos = x + (i - n_groups / 2) * width + width / 2

        ax.bar(x_pos,heights,yerr=errors,capsize=5,width=width,label=f'{group}')

    # Axis labels and ticks
    ax.set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                  labels=['Total', 'Unbouted', '<5 s', '5-10 s', '10-30 s', '30-60 s', '60-180 s', '180-600 s', '>600 s'])

    #ax.set_ylabel('Median Interday CofV (%)')
    ax.set_ylabel('Median strides / day')

    ax.set_xlabel('Bout duration')

    #ax.set_title('Median Interday CofV vs bout lengths')
    ax.set_title('Median strides / day vs bout lengths')
    ax.legend(title='Group')
    plt.tight_layout()
    plt.show()


else:
    group_medians = subj_medians.median()
    group_std = subj_std.median()


    ##############################################################################################
    #plot all bins
    # Step 2: Create the plot
    fig,axs = plt.subplots(2, figsize=(8, 9))

    axs[0].bar(group_medians.index, group_medians.values, yerr=group_med_std.values, capsize=5, color='lightblue', edgecolor='black')
    axs[0].set_title('Median unilateral steps / day')
    axs[0].set_xlabel('Bout duration (secs)')
    axs[0].set_ylabel('Unilateral steps / day')
    axs[0].set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=['Total', 'Unbouted', '<5', '5-10', '10-30', '30-60','60-180', '180-600', '>600'])

    axs[1].bar(group_CV.index, group_CV.values, yerr=group_CV_std.values, capsize=5, color='lightblue', edgecolor='black')
    axs[1].set_title('Within subject (between-day) variaiton - CV')
    axs[1].set_xlabel('Bout duration (secs)')
    axs[1].set_ylabel('CV')
    axs[1].set_xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], labels=['Total', 'Unbouted', '<5', '5-10', '10-30', '30-60','60-180', '180-600', '>600'])


    plt.tight_layout()
    plt.show()


print ('pause')

#################################################################
#CLUSTER analysis
#select subset for persposes of clustering
cluster_cols = ['not_bouted', 'short', 'medium', 'long']
subset = step1[['SUBJECT'] + cluster_cols]
subset = subset.groupby(subset['SUBJECT']).median()

#sum across subject days
print ('Run and plot cluster analysis....')
cluster_data = subset
ncluster = 4
data_out, labels = clustering(cluster_data, ncluster=ncluster)
x = np.arange(len(cluster_cols))
plt.figure(figsize=(10, 6))
sns.lineplot(data=data_out, x='feature', y='value', hue='cluster', palette='Set2')
plt.xlabel('Density (strides/min)')
plt.ylabel('Proportion of wear time')
#plt.xticks(ticks=x, labels=[cluster_cols])
plt.title('Step density pattern clusters - ALL')
plt.show()

subject_clusters = pd.DataFrame({'Subject': cluster_data.index,'Cluster': labels})

categ_table = pd.DataFrame()
cont_table = pd.DataFrame()

for i in range(ncluster):
    cluster_id = subject_clusters[subject_clusters['Cluster'] == i]
    #subjects = cluster_id['Subject']
    print ('Cluster '+str(i) + ' - n: '+ str(len(subjects)))

    cluster_demo = demo_data[demo_data['SUBJECT'].isin(cluster_id['Subject'])]
    categ, cont = create_table(cluster_demo, cont_vars, categ_vars)
    categ_table = pd.concat([categ_table, categ], ignore_index=True)
    cont_table = pd.concat([cont_table, cont], ignore_index=True)
    print ('pause')

categ_table.to_csv(summary_path+'cluster_demo_categ.csv', index=False)
cont_table.to_csv(summary_path+'cluster_demo_cont.csv', index=False)
print ('test')

#####################################################
#MEANS GRAPH
# subject means
#bouts - calculate means for bout #s
#nbout_subj_means = step2.groupby('SUBJECT')[select1].mean()
#nbout_all_means = nbout_subj_means.mean()
#nbout_all_std = nbout_subj_means.std()

#bouts setp #s absolute
nstride_subj_means = step2.groupby('SUBJECT')[select2].mean()
nstride_all_means = nstride_subj_means.mean()
nstride_all_std = nstride_subj_means.std()

#bouts setp #s percentage
nstride_pct_subj_means = step2.groupby('SUBJECT')[select3].mean()
nstride_pct_all_means = nstride_pct_subj_means.mean()
nstride_pct_all_std = nstride_pct_subj_means.std()

#coraltion matrix across bouts strides
#corr_matrix = corr_matrix_all_columns(nstride_subj_means, para=False)
#corr_matrix.to_csv(summary_path+"corr_matrix_bout_strides.csv")

'''
##################################################################
#scatter graphs

fig, axs = plt.subplots(1, 3, figsize=(10, 6))
lab_list = ['not_bouted', 'strides_>_600']
for i in lab_list:
    axs[0].scatter(nstride_subj_means['total'], nstride_subj_means[i], label=i)

lab_list = ['not_bouted_pct', 'strides_>_600_pct']
for i in lab_list:
    axs[1].scatter(nstride_subj_means['total'], nstride_pct_subj_means[i], label=i)

axs[0].set_xlabel('Total strides')
axs[0].set_ylabel('Strides')
axs[0].set_title('Strides per bout vs total strides')
axs[0].legend(loc='right')

axs[1].set_xlabel('Total strides')
axs[1].set_ylabel('% of total strides')
axs[1].set_title('% bouted strides of total')
axs[1].legend(loc='right')

# Plot stacked bars manually
bottom = [0] * len(nstride_subj_means)
x = range(len(nstride_subj_means))
# Sort rows by total row sum (ascending so largest on right)
data_sorted = nstride_subj_means.copy()
data_sorted['Total'] = nstride_subj_means.sum(axis=1)
data_sorted = data_sorted.sort_values('Total')
data_sorted = data_sorted.drop(columns='Total')

temp = data_sorted.iloc[:, 1:]
for col in temp.columns:
    axs[2].bar(x, temp[col], bottom=bottom, label=col)
    bottom = [i + j for i, j in zip(bottom, temp[col])]

plt.tight_layout()
plt.show()
print('pause')



# Step 2: Create the plot
fig,axs = plt.subplots(3, figsize=(8, 9))

axs[0].bar(nbout_all_means.index, nbout_all_means.values, yerr=nbout_all_std.values, capsize=5, color='lightgreen', edgecolor='black')
axs[0].set_title(graph_title1 + ' - Mean # bouts / day')
axs[0].set_xlabel('Bout labels')
axs[0].set_ylabel('Bouts / day')

axs[1].bar(nstride_all_means.index, nstride_all_means.values, yerr=nstride_all_std.values, capsize=5, color='lightblue', edgecolor='black')
axs[1].set_title(graph_title1 + ' - Mean # strides / day')
axs[1].set_xlabel('Bout labels')
axs[1].set_ylabel('Strides / day')

axs[2].bar(nstride_pct_all_means.index, nstride_pct_all_means.values, yerr=nstride_pct_all_std.values, capsize=5, color='violet', edgecolor='black')
axs[2].set_title(graph_title1 + ' - Mean % strides / day')
axs[2].set_xlabel('Bout labels')
axs[2].set_ylabel('Strides / day')

plt.tight_layout()
plt.show()
'''

'''
######################################################
#Day ti day varaition - coeeficitn of variation

step2_subj_cv = step2.groupby('SUBJECT')[select2].agg(lambda x: x.std() / x.mean())
step2_all_means = step2_subj_cv.mean()
step2_all_std = step2_subj_cv.std()

# Step 2: Create the plot
fig,axs = plt.subplots(3, figsize=(8, 12))

axs[0].bar(step2_all_means.index, step2_all_means.values, yerr=step2_all_std.values, capsize=5, color='lightgreen', edgecolor='black')

axs[0].set_title(graph_title1 + ' Mean COV (across days)')
axs[0].set_xlabel('Bout labels')
axs[0].set_ylabel('Mean COV (across days)')

plt.tight_layout()
plt.show()


print('pause')

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


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from Functions import read_demo_ondri_data,corr_matrix_all_columns, clustering, create_table
from scipy.stats import gaussian_kde
import seaborn as sns
import datetime

###############################################################
study = ('SA-PR01')
sub_study = 'AAIC 2025'
nimbal_dr = 'O:'
summary_path = nimbal_dr + '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\Summary_data\\'

###############################################################
#reads demographic data
new_path = 'W:\\SuperAging\\data\\summary\\'+sub_study+'\\conference\\'
file1 = study +'_collections.csv'
demodata = pd.read_csv(new_path+file1)
demodata = demodata[demodata['has_wearables_demographics'] == True]


#list of variables to tabulate
var_list = ['subject_id', 'group', 'sa_class', 'age_at_visit', 'sex', 'educ', 'mc_employment_status',
            'maristat', 'livsitua', 'independ', 'lsq_total', 'global_psqi', 'adlq_totalscore']
demodata = demodata[var_list]

##################################################################
# select certain group members
demodata = demodata[demodata['group'].isin(['control', 'superager'])].reset_index()
demodata.rename(columns={'group': 'GROUP'}, inplace=True)
demodata.rename(columns={'subject_id': 'SUBJECT'}, inplace=True)

#rename
demodata['sex'] = demodata['sex'].replace({'1': 'Male', '2': 'Female', '3':'Non-binary'})
demodata['mc_employment_status'] = demodata['mc_employment_status'].replace({1: 'Full-time', 2: 'Part-time', 3: 'Retired', 4: 'Disabled', 5: 'NA or never worked'})
demodata['maristat'] = demodata['maristat'].replace({'1': 'Married', '2': 'Widowed', '3': 'Divorced', '4': 'Separated',
                                                     '5': 'Never married', '6': 'Living as married/domestic partner', '9': 'Unknown'})
demodata['livsitua'] = demodata['livsitua'].replace({'1': 'Lives alone', '2': 'Lives (1) spouse or partner', '3': 'Lives (1) other',
                                                     '4': 'Lives with caregiver not spouse/partner, relative, or friend',
                                                     '5': 'Lives with a group private residence',
                                                     '6': 'Lives in group home', '9': 'Unknown'})
demodata['independ'] = demodata['independ'].replace({1: 'Able to live independently', 2: 'Assist-complex activities',
                                                     3: 'Assist-basic activities',
                                                     4: 'Assist-complete', 9: 'Unknown'})

#residenc
# 1, 1 Single - or multi-family private residence (apartment, condo, house)|2, 2 Retirement community or independent group living|3, 3 Assisted living, adult family home, or boarding home|4, 4 Skilled nursing facility, nursing home, hospital, or hospice|9, 9 Unknown
# demosresidenceruralurban
# ro1_nu_legacy_vars		dropdown	Residence:	1, Rural|2, Urban
#currently_exercise	participant_demographics_a1	 	yesno	Currently exercising?

# Summary calculations
categ_vars = ['GROUP', 'sex', 'mc_employment_status','maristat', 'livsitua', 'independ']
cont_vars = ['age_at_visit', 'educ', 'lsq_total', 'global_psqi', 'adlq_totalscore']
group_col = ['GROUP']
demodata[cont_vars] = demodata[cont_vars].apply(pd.to_numeric, errors='coerce')

summary_table = create_table(demodata, group_col, cont_vars, categ_vars)
summary_table = summary_table.transpose()

target_subj = demodata['SUBJECT'].unique()
subjects = pd.DataFrame(target_subj)
subjects.columns = ['SUBJECT']


#write the subject #s for the paper to file
#subjects.to_csv(summary_path +'subject_ids_'+sub_study+'.csv')

#write the subject #s for the paper to file
#demodata.to_csv(summary_path +'subject_demodata_'+sub_study+'.csv')

#write the cohort to file
summary_table.to_csv(summary_path +'subject_demo_summary_table_'+sub_study+'_v2.csv')

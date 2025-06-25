import pandas as pd


#table 1 demo detials

new_path = 'W:\\SuperAging\\data\\summary\\RPPR 2025\\SA-PR01_collections.csv'
demodata = pd.read_csv(new_path)
#list of variables to tabulate
var_list = ['study_code', 'subject_id', 'grou', 'sa_class', 'age_at_visit', 'sex','educ','mc_employment_stat', 'maristat','livsitua','independ',
            'lsq_total', 'global_psqi','adlq_score']


path = "O:\\SuperAging\\data\\summary\\AAIC 2024\\"
file1 = "SA-PR01_daily_summary.csv"
daily_sum = pd.read_csv(path+file1)

number_rows = len(daily_sum)
full_day_ankle = (daily_sum['collect_ankle'] > 1399).sum()
full_day_wrist = (daily_sum['collect_wrist'] > 1399).sum()
full_day_chest = (daily_sum['collect_chest'] > 1399).sum()

#drop rows with value of  < 1100
ankle = daily_sum[daily_sum['wear_ankle'] > 1100]
ankle['perc_wear'] = ankle['wear_ankle'] / ankle['collect_ankle']
nonwear_sums = ankle.groupby('subject_id').mean()
stats_nonwear_ankle = nonwear_sums.describe()
sum_days = ankle.groupby('subject_id').count()
stats_days_ankle = sum_days.describe()

#drop rows with value of  < 1100
arm = full_day_wrist[full_day_wrist['wear_wrist'] > 1100]
arm['perc_wear'] = arm['wear_ankle'] / arm['collect_ankle']
nonwear_sums = arm.groupby('subject_id').mean()
stats_nonwear_arm = nonwear_sums.describe()

sum_days = arm.groupby('subject_id').count()
stats_days_arm = sum_days.describe()

#drop rows with value of  < 1100
chest = full_day_chest[full_day_chest['wear_chest'] > 1100]
sum_days = chest.groupby('subject_id').count()
stats_days_chest = sum_days.describe()

sum_days = ankle.groupby('subject_id').count()
ankle['percent_wear'] = ankle['wear_ankle'] / ankle['collect_ankle']


daily_sum = daily_sum[daily_sum['collect_wrist'] > 1100]
daily_sum = daily_sum[daily_sum['collect_chest'] > 1100]


#header

''' demodata['AGE'] = demodata['age']
demodata['SUBJECT'] = demodata['subject_id']
demodata['COHORT'] = demodata['group']
demodata['EMPLOYMENT STATUS'] = None'''



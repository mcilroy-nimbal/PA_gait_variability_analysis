from Functions import wake_sleep, steps_by_day, step_density_sec,read_demo_ondri_data, read_demo_data, stride_time_interval


'''
This analysis and plotign so for Van Ootgehme 2025 paper on bouts

Uses ONDRI control data only

Part 1 of analysis

- bouts classification - medians for each grouping - by duration - includes unbouted
- Plot with

Part 2 - interday variability of bouts

'''

min_daily_hours = 20
# study = 'OND09'
study = 'SA-PR01'
# set up paths
root = 'W:'
nimbal_drive = 'O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'

#find subject list to use
demo_path = nimbal_drive+'\\Papers_NEW_April9\\Shared_Common_data\\'+study+'\\'
demodata = read_demo_data(demo_path)

if study == 'AAIC 2025':
    #subjects = pd.read_csv(summary_path + 'subject_ids_' + sub_study + '.csv')
    #master_subj_list = subjects['SUBJECT']
    print ('test')
else:
    master_subj_list = []
    for i, subject in enumerate(demodata['SUBJECT']):
        if study =='OND09':
            parts = subject.split('_', 2)  # Split into at most 3 parts
            if len(parts) == 3:
                subject = parts[0] + '_' + parts[1] + parts[2]  # Recombine without the second underscore
        elif study =='SA-PR01':
            subject='SA-PR01_'+subject

        #find non-wear to see if enough data
        # first find the Ankle, Wrist and other nonwear
        #temp = path1 + nw_path + subject + '*_NONWEAR_DAILY.csv'
        #match = glob.glob(temp)

        #ankle_list = [file for file in match if 'Ankle' in file]
        #wrist_list = [file for file in match if 'Wrist' in file]
        #chest_list = [file for file in match if 'Chest' in file]

        #if len(ankle_list) < 1:
            #continue
        #elif len(ankle_list) > 2:
            #select the first ? if there are more than 1

        master_subj_list.append(subject)


import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Functions import (wake_sleep, steps_by_day, step_density_sec,
                       read_demo_ondri_data, read_demo_data, stride_time_interval,
                       create_bin_density_files, select_subjects)
import numpy as np
import seaborn as sns
import datetime
import openpyxl
import warnings
warnings.filterwarnings("ignore")


study = 'OND09'
root = 'W:'
nimbal_drive ='O:'
paper_path = '\\Papers_NEW_April9\\In_progress\\Karen_Step_Accumulation_1\\'

#which subjects
master_subj_list = select_subjects(nimbal_drive, study)
print('Total # subjects: ' + str(len(master_subj_list)) + '\n\n')

#create sumamry data files
create_bin_density_files(study, root, nimbal_drive, paper_path, master_subj_list)
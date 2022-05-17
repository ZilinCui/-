import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil


csv_data = pd.read_csv("./objects/objects.csv")

data = np.array(csv_data)
star_list = [item[0] for item in data if item[1] == "STAR"]
galaxy_list = [item[0] for item in data if item[1] == "GALAXY"]
train_star_list, test_start_list = train_test_split(star_list, test_size=0.2, random_state=33)
train_galaxy_list, test_galaxy_list = train_test_split(galaxy_list, test_size=0.2, random_state=33)

files = os.listdir("./objects/data")
for file in files:
   path = os.path.join("./objects/data", file.lower())
   img_id = int(file.lower().replace(".jpg", ""))
   if img_id in star_list:
       if img_id in train_star_list:
           shutil.copyfile(path, f"./data/train/STAR/{img_id}.jpg")
       elif img_id in test_start_list:
           shutil.copyfile(path, f"./data/test/STAR/{img_id}.jpg")
   elif img_id in galaxy_list:
       if img_id in train_galaxy_list:
           shutil.copyfile(path, f"./data/train/GALAXY/{img_id}.jpg")
       elif img_id in test_galaxy_list:
           shutil.copyfile(path, f"./data/test/GALAXY/{img_id}.jpg")
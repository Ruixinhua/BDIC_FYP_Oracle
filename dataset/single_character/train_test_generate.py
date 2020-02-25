import os
import pandas as pd


"""This script is used to generate csv file of training and testing data with oracle inscriptions images as example"""

root_dir = "png/oracle_inscriptions/"
train_csv = "train.csv"
test_csv = "test.csv"
train_test_split = 0.8

images_names = os.listdir(root_dir)
print("The dataset contains %s images" % len(images_names))
all_frame = pd.DataFrame({"file_names": images_names, "target": [int(name.split("_")[0][1:]) for name in images_names]})
train_sample = all_frame.sample(frac=train_test_split)
test_sample = all_frame.drop(train_sample.index)
train_sample.to_csv(train_csv, index=False)
test_sample.to_csv(test_csv, index=False)

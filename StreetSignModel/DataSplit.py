import shutil
import os
import random
import pandas as pd

os.chdir("Data")

os.mkdir("StreetSigns")

shutil.unpack_archive("archive.zip", "StreetSigns/")

os.mkdir("StreetSigns/csvfiles")

shutil.move("StreetSigns/Test.csv", "StreetSigns/csvfiles")
shutil.move("StreetSigns/Train.csv", "StreetSigns/csvfiles")

shutil.move("StreetSigns/Meta.csv", "StreetSigns/csvfiles")

os.remove("archive.zip")

os.rename("StreetSigns/Train", "StreetSigns/TrainAndValid/")

if os.path.isdir('StreetSigns/Train/0/') is False:
    os.mkdir('StreetSigns/Train')
    os.mkdir('StreetSigns/Valid')

for i in range(43):
    os.mkdir(f'StreetSigns/Train/{i}')
    os.mkdir(f'StreetSigns/Valid/{i}')
    os.mkdir(f'StreetSigns/Test/{i}')

for i in range(43):

    train_samples = random.sample(os.listdir(f'StreetSigns/TrainAndValid/{i}'), 147)

    for j in train_samples:
        shutil.move(f'StreetSigns/TrainAndValid/{i}/{j}', f'StreetSigns/Train/{i}')

    valid_samples = random.sample(os.listdir(f'StreetSigns/TrainAndValid/{i}'), 63)

    for j in valid_samples:
        shutil.move(f'StreetSigns/TrainAndValid/{i}/{j}', f'StreetSigns/Valid/{i}')

Seperated_Dir = r"StreetSigns/Test/"

labels = pd.read_csv(r"StreetSigns/csvfiles/Test.csv")

for _, _, _, _, _, _, ClassID, Path in labels.values:
    shutil.move(f'StreetSigns/{Path}', f'StreetSigns/Test/{ClassID}')

shutil.rmtree("StreetSigns/TrainAndValid")

# Possible Imbalance between test classes may cause poor test performance


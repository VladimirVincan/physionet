import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def plot(series):
    min_age = series.min()
    max_age = series.max()

    plt.figure(figsize=(8,6))
    plt.hist(series, bins=range(min_age, max_age + 1), color='blue', edgecolor='black')
    plt.xlim(0,100)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def add_train_test(df, filename='RECORDS'):
    name_to_set = {}
    with open(filename, 'r') as f:
        for line in f:
            # Assuming each line is in the format 'train/name' or 'test/name'
            dataset_type = line.strip().split('/')[1]
            name = line.strip().split('/')[2]
            name_to_set[name] = dataset_type

    df['Split'] = df['Record'].map(name_to_set)
    return df

# Helper function to move files
def move_files(df, dataset_folder, dest_folder):
    # Iterate over each record and move the corresponding folder
    for record in df['Record']:
        # Define source and destination paths
        src_path = os.path.join(dataset_folder, record)
        dest_path = os.path.join(dest_folder, record)

        # Check if the folder exists
        if os.path.exists(src_path):
            # Move the folder to the destination
            shutil.move(src_path, dest_path)
        else:
            print(f"Warning: {src_path} does not exist!")


def main():
    df = pd.read_csv('age-sex.csv')
    df = add_train_test(df)
    print(df.head(-20))

    df = df[df['Split'] == 'training']

    print(df['Sex'].value_counts())
    # change the 3 rows with 'm' to 'M'
    df['Sex'] = df['Sex'].replace('m', 'M')
    # check that everything is okay
    print(df['Sex'].value_counts())

    # create age group (bins)
    bin_edges = [10, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 100]
    df['Age_group'] = pd.cut(df['Age'], bins=bin_edges, right=True)
    print(df['Age_group'].value_counts())

    # check if there are duplicates - shouldn't be any
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    print("Number of duplicate rows:", duplicates.sum())
    print(duplicate_rows)

    # plot age
    # plot(df['Age_group'])

    train, temp = train_test_split(df, test_size=200/994, stratify=df[['Sex', 'Age_group']], random_state=42)
    print(temp['Age_group'].value_counts())
    validation, test = train_test_split(temp, test_size=0.5, stratify=temp[['Sex', 'Age_group']], random_state=42)

    train.to_csv('train.csv', index=False)
    validation.to_csv('validation.csv', index=False)
    test.to_csv('test.csv', index=False)

    # ============= Splitting into subfolders =============

    dataset_folder = 'challenge-2018/training'
    train_folder = 'dataset/train'
    val_folder = 'dataset/val'
    test_folder = 'dataset/test'

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    move_files(train, dataset_folder, train_folder)
    move_files(validation, dataset_folder, val_folder)
    move_files(test, dataset_folder, test_folder)

if __name__ == '__main__':
    main()

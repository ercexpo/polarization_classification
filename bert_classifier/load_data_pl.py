# Load the dataset into a pandas dataframe.
import pandas as pd

def get_data(csv_file):
    df = pd.read_csv(csv_file, delimiter=',', header=0)

    df = df[df.polarization != 2]
    df = df[df.polarization != 9]

    df = df[(df.polarization == 1) | (df.polarization == 0)]

    df.dropna(inplace=True)

    labels = df.polarization.values
    comments = df.text.values
    print(labels)
    print(comments)
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    return comments, labels

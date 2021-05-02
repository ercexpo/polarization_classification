# Load the dataset into a pandas dataframe.
import pandas as pd

def get_data(csv_file):
    df = pd.read_csv(csv_file, delimiter=',', header=0)

    comments = df.text.values
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    return comments

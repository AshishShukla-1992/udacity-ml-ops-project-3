import pandas as pd
import numpy as np


def cleaned_data():
    # Load the data
    df = pd.read_csv('data/census_raw.csv', sep=',\s', engine='python')

    # Display the first few rows of the DataFrame to verify that it loaded correctly
    df.head()

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Strip leading and trailing spaces from string columns
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Save the cleaned DataFrame to a new CSV file
    df.to_csv('data/cleaned_data.csv', index=False)

    return df

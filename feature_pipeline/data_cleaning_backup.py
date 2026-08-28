"""
Data Cleaning Module.

Responsible for:
- Loading raw AQI data
- Removing duplicates
- Handling missing values
- Saving cleaned dataset
"""


import pandas as pd

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)


class DataCleaner:
    """
    Cleans AQI dataset.
    """


    def __init__(self):

        self.raw_file = RAW_DATA_DIR / "aqi_raw.csv"

        self.output_file = (
            PROCESSED_DATA_DIR /
            "clean_aqi.csv"
        )



    def load_data(self):

        return pd.read_csv(
            self.raw_file
        )



    def clean_data(self, df):

        # remove duplicate rows
        df = df.drop_duplicates()


        # remove missing rows
        df = df.dropna()


        # convert timestamp
        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )


        # reset index
        df = df.reset_index(
            drop=True
        )


        return df



    def save_data(self, df):

        PROCESSED_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        df.to_csv(
            self.output_file,
            index=False
        )


        print(
            f"Clean dataset saved: {self.output_file}"
        )



if __name__ == "__main__":


    cleaner = DataCleaner()


    data = cleaner.load_data()


    clean_data = cleaner.clean_data(
        data
    )


    cleaner.save_data(
        clean_data
    )


    print(clean_data.head())
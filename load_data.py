import pandas as pd

from config import (
    RAW_DATA_DIR,
)

def train_sets():
    train_df = pd.read_csv(
        RAW_DATA_DIR + 'train.csv', 
        usecols=[1, 2, 3, 4, 5],
        # converters={'unit_sales': lambda u: np.log1p(
        #     float(u)) if float(u) > 0 else 0},
        parse_dates=["date"],
        skiprows=range(1, 66458909)  # 2016-01-01
    )
    return train_df

def test_sets():
    test_df = pd.read_csv(
        RAW_DATA_DIR + 'test.csv', 
        usecols=[1, 2, 3, 4],
        # converters={'unit_sales': lambda u: np.log1p(
        #     float(u)) if float(u) > 0 else 0},
        parse_dates=["date"]
        # skiprows=range(1, 66458909)  # 2016-01-01
    )
def main():
    train_df = train_sets()
    test_df = test_sets()

    # holidays_events_df = pd.read_csv(RAW_DATA_DIR+'holidays_events.csv')
    # items_df = pd.read_csv(RAW_DATA_DIR+'holidays_events.csv')
    # oil_df = pd.read_csv(RAW_DATA_DIR+'oil.csv')
    # stores_df = pd.read_csv(RAW_DATA_DIR+'stores.csv')
    # transactions_df = pd.read_csv(RAW_DATA_DIR+'transactions.csv')

    train_df = pd.concat([
        train_df
        ])    
    return train_df, test_df


if __name__ == "__main__":
    main()
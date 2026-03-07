import pandas as pd
import pickle
from collections import defaultdict

def load_data():

    df = pd.read_excel("D:\Customer_Sequence\data\Online Retail.xlsx")

    print("Original dataset size:", df.shape)

    return df

def clean_data(df):

    # Remove rows without customer id
    df = df.dropna(subset=["CustomerID"])

    # Remove cancelled orders
    cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
    df.drop(df[cancelled].index, inplace=True)

    # Remove negative quantities
    df = df[df["Quantity"] > 0]

    df["CustomerID"] = df["CustomerID"].astype(int)

    print("Dataset after cleaning:", df.shape)

    return df

def sort_by_time(df):

    df = df.sort_values(["CustomerID", "InvoiceDate"])

    return df

def build_sequences(df):

    user_sequences = defaultdict(list)

    # group by customer + invoice (basket)
    grouped = df.groupby(["CustomerID", "InvoiceNo"])

    for (customer, invoice), group in grouped:

        # products bought in this basket
        products = list(group["StockCode"])

        # append products sequentially
        user_sequences[customer].extend(products)

    sequences = list(user_sequences.values())

    print("Total customers:", len(sequences))
    print("Example sequence:", sequences[0][:10])

    return sequences

def filter_sequences(sequences, min_length=3):

    filtered = []

    for seq in sequences:
        if len(seq) >= min_length:
            filtered.append(seq)

    print("Sequences after filtering:", len(filtered))

    return filtered

def save_sequences(sequences):

    with open("data/sequences.pkl", "wb") as f:
        pickle.dump(sequences, f)

    print("Sequences saved to data/sequences.pkl")

def main():

    df = load_data()

    df = clean_data(df)

    df = sort_by_time(df)

    sequences = build_sequences(df)

    sequences = filter_sequences(sequences)

    save_sequences(sequences)


if __name__ == "__main__":
    main()
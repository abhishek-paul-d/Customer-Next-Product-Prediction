import pickle
import torch
from sklearn.model_selection import train_test_split
from collections import Counter


def load_sequences():
    with open("data/sequences.pkl", "rb") as f:
        sequences = pickle.load(f)
    return sequences

def build_vocab(sequences, max_products=500):

    product_counts = Counter()

    for seq in sequences:
        product_counts.update(seq)

    # select most frequent products
    most_common = product_counts.most_common(max_products - 1)

    product_to_idx = {"<PAD>":0,"<UNK>": 1}
    idx_to_product = {0:"<PAD>",1: "<UNK>"}

    for i, (product, _) in enumerate(most_common, start=1):
        product_to_idx[product] = i
        idx_to_product[i] = product

    print(f"Vocabulary size (limited to {max_products}): {len(product_to_idx)}")

    return product_to_idx, idx_to_product

def encode_sequences(sequences, product_to_idx):

    encoded_sequences = []

    for seq in sequences:

        encoded_seq = [
            product_to_idx.get(product, product_to_idx["<UNK>"])
            for product in seq
        ]

        encoded_sequences.append(encoded_seq)

    print("Total sequences:", len(encoded_sequences))
    print("Example encoded sequence:", encoded_sequences[0][:10])

    return encoded_sequences

def split_sequences(encoded_sequences):

    train_val_seq, test_seq = train_test_split(
        encoded_sequences, test_size=0.2, random_state=42
    )

    train_seq, val_seq = train_test_split(
        train_val_seq, test_size=0.125, random_state=42
    )

    print("Train sequences:", len(train_seq))
    print("Validation sequences:", len(val_seq))
    print("Test sequences:", len(test_seq))

    return train_seq, val_seq, test_seq

def build_training_pairs(encoded_sequences, context_len):

    X = []
    y = []

    for seq in encoded_sequences:

        for i in range(len(seq) - context_len):

            X.append(seq[i:i + context_len])
            y.append(seq[i + context_len])

    return torch.tensor(X), torch.tensor(y)

def save_dataset(file_path, **kwargs):

    with open(file_path, "wb") as f:
        pickle.dump(kwargs, f)

def main():

    sequences = load_sequences()

    product_to_idx, idx_to_product = build_vocab(sequences, max_products=500)

    encoded_sequences = encode_sequences(sequences, product_to_idx)

    train_seq, val_seq, test_seq = split_sequences(encoded_sequences)

    # Bigram datasets
    X_bg_train, y_bg_train = build_training_pairs(train_seq, 1)
    X_bg_val, y_bg_val = build_training_pairs(val_seq, 1)
    X_bg_test, y_bg_test = build_training_pairs(test_seq, 1)

    # Trigram datasets
    X_tg_train, y_tg_train = build_training_pairs(train_seq, 2)
    X_tg_val, y_tg_val = build_training_pairs(val_seq, 2)
    X_tg_test, y_tg_test = build_training_pairs(test_seq, 2)

    # Neural model datasets
    X_nn_train, y_nn_train = build_training_pairs(train_seq, 3)
    X_nn_val, y_nn_val = build_training_pairs(val_seq, 3)
    X_nn_test, y_nn_test = build_training_pairs(test_seq, 3)

    print("\nDataset Sizes")
    print("Bigram train:", X_bg_train.shape)
    print(f"Neural Net Train Shape",X_nn_train.shape)
    print(f"Trigram Train Shape",X_tg_train.shape)

    save_dataset(
        "data/dataset_all.pkl",

        X_bg_train=X_bg_train, y_bg_train=y_bg_train,
        X_bg_val=X_bg_val, y_bg_val=y_bg_val,
        X_bg_test=X_bg_test, y_bg_test=y_bg_test,

        X_tg_train=X_tg_train, y_tg_train=y_tg_train,
        X_tg_val=X_tg_val, y_tg_val=y_tg_val,
        X_tg_test=X_tg_test, y_tg_test=y_tg_test,

        X_nn_train=X_nn_train, y_nn_train=y_nn_train,
        X_nn_val=X_nn_val, y_nn_val=y_nn_val,
        X_nn_test=X_nn_test, y_nn_test=y_nn_test,

        product_to_idx=product_to_idx,
        idx_to_product=idx_to_product
    )
    print("\nDataset built and saved successfully!")


if __name__ == "__main__":
    main()
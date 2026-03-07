import pickle
import os
import torch
from src.models.bigram import BigramModel


def load_dataset():
    with open("data/dataset_all.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def save_model(model, path="models/bigram.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

def main():

    data = load_dataset()

    X_train = data["X_bg_train"]
    y_train = data["y_bg_train"]

    X_val = data["X_bg_val"]
    y_val = data["y_bg_val"]

    X_test=data["X_bg_test"]
    y_test=data["y_bg_test"]

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.long)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train, dtype=torch.long)

    if not isinstance(X_val, torch.Tensor):
        X_val = torch.tensor(X_val, dtype=torch.long)

    if not isinstance(y_val, torch.Tensor):
        y_val = torch.tensor(y_val, dtype=torch.long)

    vocab_size = len(data["product_to_idx"])

    print("Training Bigram Model...")
    model = BigramModel(vocab_size=vocab_size)

    model.train(X_train, y_train)

    # Train metrics
    train_ppl = model.perplexity(X_train, y_train)
    train_acc = model.topk_accuracy(X_train, y_train, k=5)

    print("Train Perplexity:", train_ppl)
    print("Train Top-5 Accuracy:", train_acc)

    # Validation metrics
    val_ppl = model.perplexity(X_val, y_val)
    val_acc = model.topk_accuracy(X_val, y_val, k=5)

    print("Validation Perplexity:", val_ppl)
    print("Validation Top-5 Accuracy:", val_acc)

    test_ppl=model.perplexity(X_test,y_test)
    test_acc=model.topk_accuracy(X_test,y_test)
    print(f"Test Perplexity: {test_ppl:.4f}")
    print(f"Test Top-5 Accuracy: {test_acc:.4f}")

    # Example prediction
    first_idx = X_train[0][0].item()
    top5_next=model.get_topk_predictions(first_idx,data["idx_to_product"],k=5)

    for r in top5_next:
        print(f"Rank {r['rank']}: Product {r['product_name']} (idx={r['product_idx']}), Prob={r['probability']:.4f}")

    save_model(model)

    print("Bigram model trained and saved successfully!")

if __name__ == "__main__":
    main()
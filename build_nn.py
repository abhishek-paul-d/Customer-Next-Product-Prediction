import pickle
import os
import torch
from src.models.NeuralNet import NeuralNet

def load_data():
    with open("data/dataset_all.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def save_model(model,path='models/nn_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def main():

    data=load_data()
    X_train=data["X_nn_train"]
    y_train=data["y_nn_train"]

    X_val=data["X_nn_val"]
    y_val=data["y_nn_val"]

    X_test=data["X_nn_test"]
    y_test=data["y_nn_test"]

    if not isinstance(X_train, torch.Tensor):
        X_train=torch.tensor(X_train,dtype=torch.long)
    if not isinstance(y_train, torch.Tensor):
        y_train=torch.tensor(y_train,dtype=torch.long)  

    if not isinstance(X_val, torch.Tensor):
        X_val=torch.tensor(X_val,dtype=torch.long)
    if not isinstance(y_val, torch.Tensor):
        y_val=torch.tensor(y_val,dtype=torch.long)

    vocab_size=len(data["product_to_idx"])
    print("Training Neural Network Model...")
    model=NeuralNet(vocab_size)
    model.train_model(X_train,y_train,epochs=20)
    train_ppl=model.perplexity(X_train,y_train)
    train_acc=model.topk_accuracy(X_train,y_train,k=5)
    val_ppl=model.perplexity(X_val,y_val)
    val_acc=model.topk_accuracy(X_val,y_val,k=5)
    test_ppl=model.perplexity(X_test,y_test)
    test_acc=model.topk_accuracy(X_test,y_test,k=5)
    
    
    print(f"Train Perplexity: {train_ppl:.4f}, Train Top-5 Accuracy: {train_acc:.4f}")
    print(f"Validation Perplexity: {val_ppl:.4f}, Validation Top-5 Accuracy: {val_acc:.4f}")
    print(f"Test Perplexity: {test_ppl:.4f}, Test Top-5 Accuracy: {test_acc:.4f}")

    sample = X_train[0].unsqueeze(0)
    pred = model.get_topk_predictions(sample, data["idx_to_product"])
    for r in pred:
        print(f"Rank {r['rank']}: Product {r['product_name']} (idx={r['product_idx']}), Prob={r['probability']:.4f}")

    
    save_model(model)
    print("Model saved to models/nn_model.pkl")

if __name__=="__main__":
    main()




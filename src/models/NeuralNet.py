import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, context_size=3, dropout=0.4):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Hidden layers
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)

        h = torch.relu(self.fc1(emb))
        h = self.dropout(h)

        out = self.fc2(h)
        return out

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------

    def train_model(self, X_train, y_train, epochs=20, batch_size=256, learning_rate=0.001):

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate,weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        dataset_size = len(X_train)

        for epoch in range(epochs):

            self.train()

            # Shuffle dataset
            perm = torch.randperm(dataset_size)
            X_train = X_train[perm]
            y_train = y_train[perm]

            total_loss = 0

            for i in range(0, dataset_size, batch_size):

                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                optimizer.zero_grad()

                logits = self.forward(xb)
                loss = loss_fn(logits, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            avg_loss = total_loss / (dataset_size / batch_size)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # ---------------------------------------------
    # PERPLEXITY
    # ---------------------------------------------

    def perplexity(self, X, y, batch_size=512):

        self.eval()

        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():

            for i in range(0, len(X), batch_size):

                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                logits = self.forward(xb)

                loss = loss_fn(logits, yb)

                total_loss += loss.item()
                total_tokens += len(yb)

        avg_loss = total_loss / total_tokens
        ppl = torch.exp(torch.tensor(avg_loss))

        return ppl.item()

    # ---------------------------------------------
    # TOP-K ACCURACY
    # ---------------------------------------------

    def topk_accuracy(self, X, y, k=5, batch_size=512):

        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for i in range(0, len(X), batch_size):

                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                logits = self.forward(xb)

                topk = torch.topk(logits, k).indices

                for j in range(len(yb)):
                    if yb[j] in topk[j]:
                        correct += 1

                total += len(yb)

        return correct / total

    # ---------------------------------------------
    # PREDICTION
    # ---------------------------------------------

    def predict_next(self, x):

        self.eval()

        with torch.no_grad():

            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)

        return probs

    def next_prediction(self, x):

        probs = self.predict_next(x)

        return torch.argmax(probs, dim=-1)

    def top_k(self, x, k=5):

        probs = self.predict_next(x)

        values, indices = torch.topk(probs, k)

        return indices, values

    def get_topk_predictions(self, x, idx_to_product, k=5):

        indices, values = self.top_k(x, k)

        results = []

        for i in range(k):

            pred_idx = indices[0][i].item()
            pred_product = idx_to_product.get(pred_idx, "<UNK>")
            prob = values[0][i].item()

            results.append({
                "rank": i + 1,
                "product_idx": pred_idx,
                "product_name": pred_product,
                "probability": prob
            })

        return results
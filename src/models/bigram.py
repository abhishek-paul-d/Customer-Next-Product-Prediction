import torch

class BigramModel:

    def __init__(self, vocab_size, discount=0.75):
        self.vocab_size = vocab_size
        self.discount = discount

        self.counts = torch.zeros((vocab_size, vocab_size))  
        self.context_counts = torch.zeros(vocab_size)     
        self.continuation_counts = torch.zeros(vocab_size)
        self.probs = None

    def train(self, X, y):
        for i in range(len(X)):
            prev = X[i][0].item()
            nxt = y[i].item()
            self.counts[prev][nxt] += 1
            self.context_counts[prev] += 1

        for i in range(self.vocab_size):
            self.continuation_counts[i] = torch.sum(self.counts[:, i] > 0)

        total_continuations = torch.sum(self.continuation_counts)
        continuation_probs = self.continuation_counts / total_continuations

        probs = torch.zeros_like(self.counts)
        for prev in range(self.vocab_size):
            context_total = self.context_counts[prev]

            if context_total == 0:
                continue

            unique_continuations = torch.sum(self.counts[prev] > 0)
            lambda_weight = (self.discount * unique_continuations) / context_total

            for nxt in range(self.vocab_size):
                count = self.counts[prev, nxt]
                discounted = max(count - self.discount, 0) / context_total
                probs[prev, nxt] = discounted + lambda_weight * continuation_probs[nxt]

        unigram_probs = self.context_counts / torch.sum(self.context_counts)
        self.probs = torch.zeros_like(self.counts)
        for prev in range(self.vocab_size):
            for nxt in range(self.vocab_size):
                if self.counts[prev, nxt] > 0:
                    self.probs[prev, nxt] = probs[prev, nxt]
                else:
                    self.probs[prev, nxt] = unigram_probs[nxt]

    def predict_next(self, product_idx):
        return self.probs[product_idx]

    def top_k(self, product_idx, k=5):
        probs = self.predict_next(product_idx)
        values, indices = torch.topk(probs, k)
        return indices, values

    def perplexity(self, X, y):
        log_probs = 0.0
        N = len(X)
        for i in range(N):
            prev = X[i][0].item()
            nxt = y[i].item()
            p = self.probs[prev, nxt].item()
            p = max(p, 1e-10)  # Avoid log(0)
            log_probs += torch.log(torch.tensor(p))
        avg_log_prob = log_probs / N
        ppl = torch.exp(-avg_log_prob)
        return ppl.item()

    def topk_accuracy(self, X, y, k=5):
        correct = 0
        for i in range(len(X)):
            prev = X[i][0].item()
            nxt = y[i].item()
            top_indices, _ = self.top_k(prev, k)
            if nxt in top_indices:
                correct += 1
        return correct / len(X)

    def get_topk_predictions(self, product_idx, idx_to_product, k=5):
        probs = self.predict_next(product_idx)
        values, indices = torch.topk(probs, k)

        indices = indices.squeeze().tolist()
        values = values.squeeze().tolist()

        print(f"\nTop-{k} predictions after product {product_idx}:")
        results = []
        for i in range(k):
            pred_idx = indices[i]
            pred_product = idx_to_product.get(pred_idx, "<UNK>")
            prob = values[i]

            results.append({
                "rank": i + 1,
                "product_idx": pred_idx,
                "product_name": pred_product,
                "probability": prob
            })

        return results

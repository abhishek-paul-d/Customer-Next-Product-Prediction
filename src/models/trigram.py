import torch
from collections import defaultdict


class TrigramModel:

    def __init__(self, vocab_size, discount=0.75):
        self.vocab_size = vocab_size
        self.discount = discount

        self.trigram_counts = defaultdict(int)
        self.bigram_context_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        self.unique_followers = defaultdict(set)

    def train(self, X, y):
        for i in range(len(X)):
            w1 = int(X[i][0])
            w2 = int(X[i][1])
            w3 = int(y[i])

            self.trigram_counts[(w1, w2, w3)] += 1
            self.bigram_context_counts[(w1, w2)] += 1
            self.unigram_counts[w3] += 1
            self.unique_followers[(w1, w2)].add(w3)

    def prob(self, w1, w2, w3):
        w1 = int(w1)
        w2 = int(w2)
        w3 = int(w3)

        trigram_count = self.trigram_counts[(w1, w2, w3)]
        bigram_count = self.bigram_context_counts[(w1, w2)]
        unigram_count = self.unigram_counts[w3]

        if bigram_count == 0:
            unigram_total = sum(self.unigram_counts.values())
            return max(unigram_count - self.discount, 0) / unigram_total

        if trigram_count > 0:
            discounted = max(trigram_count - self.discount, 0) / bigram_count
            unique_continuations = len(self.unique_followers[(w1, w2)])
            lambda_weight = (self.discount * unique_continuations) / bigram_count

            return discounted + lambda_weight * (unigram_count / sum(self.unigram_counts.values()))

        lambda_weight = (self.discount * len(self.unique_followers[(w1, w2)])) / bigram_count
        backoff_prob = max(bigram_count - self.discount, 0) / bigram_count
        return backoff_prob + lambda_weight * (unigram_count / sum(self.unigram_counts.values()))

    def predict_next(self, w1, w2):
        probs = torch.zeros(self.vocab_size)

        for w3 in range(self.vocab_size):
            probs[w3] = self.prob(w1, w2, w3)

        return probs

    def top_k(self, w1, w2, k=5):
        probs = self.predict_next(w1, w2)

        values, indices = torch.topk(probs, k)

        return indices, values

    def perplexity(self, X, y):
        log_prob_sum = 0.0
        N = len(X)
        for i in range(N):
            w1 = int(X[i][0])
            w2 = int(X[i][1])
            w3 = int(y[i])

            p = self.prob(w1, w2, w3)
            p = max(p, 1e-10)

            log_prob_sum += torch.log(torch.tensor(p))

        ppl = torch.exp(-log_prob_sum / N)

        return ppl.item()

    def topk_accuracy(self, X, y, k=5):
        correct = 0
        N = len(X)

        for i in range(N):
            w1 = int(X[i][0])
            w2 = int(X[i][1])
            true_word = int(y[i])
            probs = self.predict_next(w1, w2)
            topk = torch.topk(probs, k).indices.tolist()
            if true_word in topk:
                correct += 1

        return correct / N

    def get_topk_predictions(self, product_idx1, product_idx2, idx_to_product, k=5):
        x1 = torch.tensor([product_idx1])
        x2 = torch.tensor([product_idx2])
        probs = self.predict_next(x1, x2)
        values, indices = torch.topk(probs, k)

        indices = indices.squeeze().tolist()
        values = values.squeeze().tolist()

        print(f"\nTop-{k} predictions after product {product_idx1} and {product_idx2}:")
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
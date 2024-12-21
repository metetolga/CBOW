
# Word Embedding - CBOW
import re
import nltk
import numpy as np
from matplotlib import pyplot
from nltk.tokenize import word_tokenize
from utils import get_batches, compute_pca, get_dict

# Append NLTK data path
nltk.data.path.append('.')

# Load and preprocess data
def preprocess_data(file_path):
    with open(file_path) as f:
        data = f.read()
    data = re.sub(r'[,!?;-]', '.', data)
    tokens = nltk.word_tokenize(data)
    tokens = [ch.lower() for ch in tokens if ch.isalpha() or ch == '.']
    return tokens

data = preprocess_data('shakespeare.txt')
print("Number of tokens:", len(data), '\n', data[:15])

# Frequency distribution and vocabulary creation
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary: ", len(fdist))
print("Most frequent tokens: ", fdist.most_common(20))

word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
print("Size of vocabulary: ", V)

# Example usage of dictionaries
print("Index of the word 'king':", word2Ind.get('king', 'Not found'))
print("Word at index 2743:", Ind2word.get(2743, 'Not found'))

# Model Initialization
def init_model(N, V, random_seed=42):
    np.random.seed(random_seed)
    W1 = np.random.rand(N, V)
    b1 = np.random.rand(N, 1)
    W2 = np.random.rand(V, N)
    b2 = np.random.rand(V, 1)
    return W1, W2, b1, b2

# Softmax function
def softmax(z):
    z_exp = np.exp(z)
    z_exp_colsum = np.sum(z_exp, axis=0)
    return z_exp / z_exp_colsum

# Forward propagation
def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1, x) + b1
    h = np.maximum(0, h)  # ReLU activation
    z = np.dot(W2, h) + b2
    return z, h

# Compute cost
def compute_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), 1 - y)
    cost = -1 / batch_size * np.sum(logprobs)
    return np.squeeze(cost)

# Backpropagation
def back_prop(x, yhat, y, h, W1, W2, batch_size):
    l1 = np.dot(W2.T, (yhat - y))
    l1 = np.maximum(0, l1)  # Derivative of ReLU
    grad_W1 = (1 / batch_size) * np.dot(l1, x.T)
    grad_W2 = (1 / batch_size) * np.dot(yhat - y, h.T)
    grad_b1 = (1 / batch_size) * np.sum(l1, axis=1, keepdims=True)
    grad_b2 = (1 / batch_size) * np.sum(yhat - y, axis=1, keepdims=True)
    return grad_W1, grad_W2, grad_b1, grad_b2

# Gradient Descent
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    W1, W2, b1, b2 = init_model(N, V, random_seed=282)
    batch_size = 128
    C = 2
    for iters, (x, y) in enumerate(get_batches(data, word2Ind, V, C, batch_size)):
        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = compute_cost(y, yhat, batch_size)

        if (iters + 1) % 10 == 0:
            print(f"Iteration {iters + 1}, Cost: {cost:.6f}")

        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, batch_size)
        W1 -= alpha * grad_W1
        W2 -= alpha * grad_W2
        b1 -= alpha * grad_b1
        b2 -= alpha * grad_b2

        if (iters + 1) % 100 == 0:
            alpha *= 0.66
        if iters + 1 == num_iters:
            break

    return W1, W2, b1, b2

# Visualize embeddings
def visualize_embeddings(W1, W2, word2Ind, words, n_components):
    embs = (W1.T + W2) / 2.0
    idx = [word2Ind[word] for word in words if word in word2Ind]
    X = embs[idx, :]

    result = compute_pca(X, n_components)
    pyplot.scatter(result[:, (n_components-1)], result[:, 1])
    for i, word in enumerate(words):
        if word in word2Ind:
            pyplot.annotate(word, xy=(result[i, (n_components-1)], result[i, 1]))
    pyplot.show()

if __name__ == "__main__":
    # Train the model
    C = 2
    N = 50
    num_iters = 150
    print("Training the model...")
    W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)

    words = ['king', 'queen', 'lord', 'man', 'woman', 'dog', 'horse', 'rich', 'happy', 'sad']
    visualize_embeddings(W1, W2, word2Ind, words, 5)
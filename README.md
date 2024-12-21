# CBOW Word Embedding Model

This repository contains an implementation of the Continuous Bag of Words (CBOW) model for learning word embeddings. The code processes text data, builds a vocabulary, trains the CBOW model, and visualizes word embeddings.

## Table of Contents

1. [Setup and Preprocessing](#setup-and-preprocessing)
2. [Functions Overview](#functions-overview)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Initialization](#model-initialization)
   - [Forward Propagation](#forward-propagation)
   - [Cost Computation](#cost-computation)
   - [Backpropagation](#backpropagation)
   - [Gradient Descent](#gradient-descent)
   - [Embedding Visualization](#embedding-visualization)
3. [Usage](#usage)

## Setup and Preprocessing

The dataset is preprocessed to remove unnecessary characters and tokenize words. Frequency distribution and dictionaries (`word2Ind` and `Ind2word`) are created to manage the vocabulary.

```python
# Example usage
word2Ind, Ind2word = get_dict(data)
print("Index of 'king':", word2Ind.get('king', 'Not found'))
print("Word at index 2743:", Ind2word.get(2743, 'Not found'))
```

## Functions Overview

### Data Preprocessing

`preprocess_data(file_path)`
- **Input**: Path to the text file.
- **Output**: List of tokens with punctuation removed and all words in lowercase.

```python
def preprocess_data(file_path):
    with open(file_path) as f:
        data = f.read()
    data = re.sub(r'[,!?;-]', '.', data)  # Replace punctuation with periods
    tokens = nltk.word_tokenize(data)  # Tokenize the text
    tokens = [ch.lower() for ch in tokens if ch.isalpha() or ch == '.']  # Convert to lowercase and keep valid tokens
    return tokens
```

### Model Initialization

`init_model(N, V, random_seed=42)`
- **Input**: Embedding dimension `N`, vocabulary size `V`, and a random seed.
- **Output**: Randomly initialized weights and biases (`W1`, `W2`, `b1`, `b2`).

```python
def init_model(N, V, random_seed=42):
    np.random.seed(random_seed)
    W1 = np.random.rand(N, V)  # Initialize W1 randomly
    b1 = np.random.rand(N, 1)  # Initialize b1 randomly
    W2 = np.random.rand(V, N)  # Initialize W2 randomly
    b2 = np.random.rand(V, 1)  # Initialize b2 randomly
    return W1, W2, b1, b2
```

### Forward Propagation

`forward_prop(x, W1, W2, b1, b2)`
- **Input**: One-hot encoded input `x` and model parameters (`W1`, `W2`, `b1`, `b2`).
- **Output**: 
  - `z`: Output scores for all vocabulary words.
  - `h`: Hidden layer activations.

```python
def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1, x) + b1  # Compute hidden layer
    h = np.maximum(0, h)  # Apply ReLU activation
    z = np.dot(W2, h) + b2  # Compute output scores
    return z, h
```

### Cost Computation

`compute_cost(y, yhat, batch_size)`
- **Input**: True labels `y`, predicted probabilities `yhat`, and batch size.
- **Output**: Scalar cost value computed using cross-entropy loss.

```python
def compute_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat), y) + np.multiply(np.log(1 - yhat), 1 - y)  # Compute log probabilities
    cost = -1 / batch_size * np.sum(logprobs)  # Compute average cross-entropy loss
    return np.squeeze(cost)
```

### Backpropagation

`back_prop(x, yhat, y, h, W1, W2, batch_size)`
- **Input**: Input `x`, predicted probabilities `yhat`, true labels `y`, hidden layer activations `h`, and model parameters.
- **Output**: Gradients for weights (`grad_W1`, `grad_W2`) and biases (`grad_b1`, `grad_b2`).

```python
def back_prop(x, yhat, y, h, W1, W2, batch_size):
    l1 = np.dot(W2.T, (yhat - y))  # Compute gradient of loss w.r.t hidden layer
    l1 = np.maximum(0, l1)  # Apply ReLU derivative
    grad_W1 = (1 / batch_size) * np.dot(l1, x.T)  # Compute gradient for W1
    grad_W2 = (1 / batch_size) * np.dot(yhat - y, h.T)  # Compute gradient for W2
    grad_b1 = (1 / batch_size) * np.sum(l1, axis=1, keepdims=True)  # Compute gradient for b1
    grad_b2 = (1 / batch_size) * np.sum(yhat - y, axis=1, keepdims=True)  # Compute gradient for b2
    return grad_W1, grad_W2, grad_b1, grad_b2
```

### Gradient Descent

`gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03)`
- **Input**: Tokenized data, word-to-index dictionary `word2Ind`, embedding size `N`, vocabulary size `V`, number of iterations `num_iters`, and learning rate `alpha`.
- **Output**: Trained weights and biases.
- **Key Features**: Adaptive learning rate reduces every 100 iterations.

```python
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    W1, W2, b1, b2 = init_model(N, V, random_seed=282)  # Initialize model
    batch_size = 128
    C = 2  # Context size
    for iters, (x, y) in enumerate(get_batches(data, word2Ind, V, C, batch_size)):
        z, h = forward_prop(x, W1, W2, b1, b2)  # Perform forward propagation
        yhat = softmax(z)  # Compute softmax probabilities
        cost = compute_cost(y, yhat, batch_size)  # Compute cost

        if (iters + 1) % 10 == 0:
            print(f"Iteration {iters + 1}, Cost: {cost:.6f}")

        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, batch_size)  # Compute gradients
        W1 -= alpha * grad_W1  # Update W1
        W2 -= alpha * grad_W2  # Update W2
        b1 -= alpha * grad_b1  # Update b1
        b2 -= alpha * grad_b2  # Update b2

        if (iters + 1) % 100 == 0:
            alpha *= 0.66  # Reduce learning rate
        if iters + 1 == num_iters:
            break

    return W1, W2, b1, b2
```

### Embedding Visualization

`visualize_embeddings(W1, W2, word2Ind, words)`
- **Input**: Trained weights, `word2Ind` dictionary, and list of target words.
- **Output**: 2D PCA plot of selected word embeddings.

```python
def visualize_embeddings(W1, W2, word2Ind, words):
    embs = (W1.T + W2) / 2.0  # Compute average embedding
    idx = [word2Ind[word] for word in words if word in word2Ind]  # Get indices of target words
    X = embs[idx, :]  # Extract embeddings

    result = compute_pca(X, 2)  # Compute 2D PCA
    pyplot.scatter(result[:, 0], result[:, 1])  # Plot embeddings
    for i, word in enumerate(words):
        if word in word2Ind:
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))  # Annotate words
    pyplot.show()
```

## Usage

1. Preprocess the data and create vocabulary dictionaries.
2. Initialize and train the CBOW model using gradient descent.
3. Visualize the learned embeddings.

### Training Example

```python
C = 2  # Context size
N = 50  # Embedding size
num_iters = 150  # Number of iterations
print("Training the model...")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)
```


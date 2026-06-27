import random
import math

# ==========================================
# 1. Training Corpus & Tokenization
# ==========================================
CORPUS = """
def gator_sort(arr):
    print("Codegator is sorting your data")
    return sorted(arr)

def codegator_status():
    print("Codegator AI is awake and running smoothly")

# AI training examples
codegator code is simple and clean
machine learning code from scratch is powerful
hello human welcome to the codegator terminal
"""

# Clean and split the text into distinct word tokens
words = CORPUS.lower().split()
unique_words = sorted(list(set(words)))
vocab_size = len(unique_words)

# Map words to numbers and vice versa
word_to_idx = {word: i for i, word in enumerate(unique_words)}
idx_to_word = {i: word for i, word in enumerate(unique_words)}

print(f"🐊 [Codegator Deep MLP Initialized]")
print(f"Vocabulary Size: {vocab_size} unique words digested.")

# ==========================================
# 2. Hyperparameters & Network Weights
# ==========================================
hidden_size = 16  # Dimensionality of the hidden state
learning_rate = 0.1
epochs = 2000

# Xavier/Glorot-like initialization for weights from scratch
def init_matrix(rows, cols):
    bound = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]

def init_vector(size):
    return [0.0 for _ in range(size)]

# Network Weights and Biases
# W1: Input -> Hidden (Since input is 1-hot, W1[idx] acts as the embedding for that word)
W1 = init_matrix(vocab_size, hidden_size)
b1 = init_vector(hidden_size)

# W2: Hidden -> Output
W2 = init_matrix(hidden_size, vocab_size)
b2 = init_vector(vocab_size)

# ==========================================
# 3. Activation Functions
# ==========================================
def tanh(vector):
    return [math.tanh(x) for x in vector]

def softmax(vector):
    max_val = max(vector)
    exp_vector = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vector)
    return [x / sum_exp for x in exp_vector]

# ==========================================
# 4. Neural Network Training Phase
# ==========================================
print(f"🐊 Codegator is training its deep network (Hidden Size: {hidden_size})...")

for epoch in range(epochs):
    epoch_loss = 0
    
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i+1]
        
        input_idx = word_to_idx[current_word]
        target_idx = word_to_idx[next_word]
        
        # --- FORWARD PASS ---
        # Layer 1: Input to Hidden
        # Because input is a one-hot vector, multiplying it by W1 is identical to slicing W1[input_idx]
        hidden_raw = [W1[input_idx][h] + b1[h] for h in range(hidden_size)]
        hidden_activated = tanh(hidden_raw)
        
        # Layer 2: Hidden to Output
        output_raw = [b2[out] for out in range(vocab_size)]
        for out in range(vocab_size):
            for h in range(hidden_size):
                output_raw[out] += hidden_activated[h] * W2[h][out]
                
        probabilities = softmax(output_raw)
        
        # Track Categorical Cross-Entropy Loss
        epoch_loss += -math.log(max(probabilities[target_idx], 1e-15))
        
        # --- BACKWARD PASS (Chain Rule from Scratch) ---
        # 1. Output Error Gradient (dE/d_output_raw)
        d_output = [prob for prob in probabilities]
        d_output[target_idx] -= 1.0  # Softmax + Cross-Entropy gradient simplifies beautifully to (prob - target)
        
        # 2. Gradients for Layer 2 Weights (W2) and Biases (b2)
        dW2 = [[0.0 for _ in range(vocab_size)] for _ in range(hidden_size)]
        db2 = [grad for grad in d_output]
        
        for h in range(hidden_size):
            for out in range(vocab_size):
                dW2[h][out] = hidden_activated[h] * d_output[out]
                
        # 3. Backprop error to Hidden Layer (dE/d_hidden_activated)
        d_hidden_activated = [0.0 for _ in range(hidden_size)]
        for h in range(hidden_size):
            for out in range(vocab_size):
                d_hidden_activated[h] += d_output[out] * W2[h][out]
                
        # 4. Backprop through Tanh non-linearity: d_tanh = 1 - tanh(x)^2
        d_hidden_raw = [d_hidden_activated[h] * (1.0 - hidden_activated[h] ** 2) for h in range(hidden_size)]
        
        # 5. Gradients for Layer 1 Weights (W1) and Biases (b1)
        # Note: Only the row corresponding to input_idx in W1 receives a gradient update
        dW1_row = [grad for grad in d_hidden_raw]
        db1 = [grad for grad in d_hidden_raw]
        
        # --- GRADIENT DESCENT WEIGHT UPDATE ---
        # Update Layer 2
        for out in range(vocab_size):
            b2[out] -= learning_rate * db2[out]
            for h in range(hidden_size):
                W2[h][out] -= learning_rate * dW2[h][out]
                
        # Update Layer 1
        for h in range(hidden_size):
            b1[h] -= learning_rate * db1[h]
            W1[input_idx][h] -= learning_rate * dW1_row[h]

    # Print diagnostics
    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:04d}/{epochs} | Cross-Entropy Loss: {epoch_loss:.4f}")

print("🐊 Training complete! Codegator's hidden layer representations are fully calibrated.\n")

# ==========================================
# 5. Text Generation Function
# ==========================================
def codegator_respond(start_word, num_words=10):
    start_word = start_word.lower().strip()
    if start_word not in word_to_idx:
        start_word = random.choice(unique_words)
        
    current_word = start_word
    output = [current_word]
    
    for _ in range(num_words):
        input_idx = word_to_idx[current_word]
        
        # Forward pass inference
        hidden_raw = [W1[input_idx][h] + b1[h] for h in range(hidden_size)]
        hidden_activated = tanh(hidden_raw)
        
        output_raw = [b2[out] for out in range(vocab_size)]
        for out in range(vocab_size):
            for h in range(hidden_size):
                output_raw[out] += hidden_activated[h] * W2[h][out]
                
        probabilities = softmax(output_raw)
        
        # Weighted random choice sampling
        r = random.random()
        cumulative = 0.0
        next_idx = 0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                next_idx = idx
                break
                
        current_word = idx_to_word[next_idx]
        output.append(current_word)
        
    return " ".join(output)

# ==========================================
# 6. Interactive Loop
# ==========================================
print("--- Chat with Deep Codegator ---")
print("Type a single word to seed Codegator's text generation (or type 'exit')\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("🐊 Codegator goes back to sleep. Goodbye!")
            break
        if not user_input:
            continue
            
        first_word = user_input.split()[0]
        response = codegator_respond(first_word, num_words=12)
        print(f"Codegator: ... {response} ...\n")
        
    except KeyboardInterrupt:
        print("\n🐊 Codegator out!")
        break

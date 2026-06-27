import random
import math

# 1. Training Corpus (The text the AI learns from)
# You can replace this with any text you want your AI to study.
CORPUS = """
hello world hello human hello artificial intelligence
welcome to the machine learning code from scratch
artificial intelligence is beautiful and simple
human loves machine learning and code
"""

# 2. The Tokenizer (Turns letters/characters into numbers)
# A computer cannot read characters, so we map unique characters to indexes.
chars = sorted(list(set(CORPUS)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"--- AI Initialization ---")
print(f"Vocabulary Size: {vocab_size} unique characters.")

# 3. Neural Weights Setup (The AI's Brain Matrix)
# We initialize a matrix of weights filled with small random numbers.
# Size is vocab_size x vocab_size (Transition probabilities)
weights = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(vocab_size)]

# 4. Activation Function (Softmax)
# Converts raw matrix numerical scores into stable probability percentages (0 to 1)
def softmax(vector):
    # Subtract max value for numerical stability (prevents overflow errors)
    max_val = max(vector)
    exp_vector = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vector)
    return [x / sum_exp for x in exp_vector]

# 5. Training Loop (Gradient Descent optimization from scratch)
print("\nTraining the AI network...")
learning_rate = 0.1
epochs = 5000

for epoch in range(epochs):
    loss = 0
    # Process the text sequence pairing current character -> next character
    for i in range(len(CORPUS) - 1):
        current_char = CORPUS[i]
        next_char = CORPUS[i+1]
        
        input_idx = char_to_idx[current_char]
        target_idx = char_to_idx[next_char]
        
        # Forward Pass: Extract the weight row for the current character input
        raw_scores = weights[input_idx]
        probabilities = softmax(raw_scores)
        
        # Calculate loss (Cross-Entropy Loss metric)
        loss += -math.log(max(probabilities[target_idx], 1e-15))
        
        # Backward Pass / Gradient calculation:
        # Gradient = Probability output minus Target (1 for the correct character, 0 otherwise)
        for j in range(vocab_size):
            target_value = 1.0 if j == target_idx else 0.0
            gradient = probabilities[j] - target_value
            
            # Update the weight adjustments directly
            weights[input_idx][j] -= learning_rate * gradient

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

print("Training Complete! The AI has calibrated its matrix.")

# 6. Inference Engine (Generating Text)
def generate_text(start_char, length=40):
    current_char = start_char
    if current_char not in char_to_idx:
        current_char = CORPUS[0]
        
    output = current_char
    
    for _ in range(length):
        input_idx = char_to_idx[current_char]
        
        # Query weights and turn them into mathematical choices
        raw_scores = weights[input_idx]
        probabilities = softmax(raw_scores)
        
        # Standard random weighted choice selection
        r = random.random()
        cumulative = 0.0
        next_idx = 0
        for idx, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                next_idx = idx
                break
                
        current_char = idx_to_char[next_idx]
        output += current_char
        
    return output

# Run generation tests
print("\n--- AI Generation Output ---")
for seed in ['h', 'a', 'm']:
    print(f"Seed '{seed}' output -> {generate_text(seed, 45)}")

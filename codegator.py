import random
import math

# 1. Training Corpus (What Codegator eats to learn)
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

print(f"🐊 [Codegator Initialized]")
print(f"Vocabulary Size: {vocab_size} unique words digested.")

# 2. Initialize Codegator's Brain Matrix (Weights)
weights = [[random.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(vocab_size)]

def softmax(vector):
    max_val = max(vector)
    exp_vector = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vector)
    return [x / sum_exp for x in exp_vector]

# 3. The Training Phase
print("\n🐊 Codegator is training its neural weights...")
learning_rate = 0.2
epochs = 3000

for epoch in range(epochs):
    loss = 0
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i+1]
        
        input_idx = word_to_idx[current_word]
        target_idx = word_to_idx[next_word]
        
        # Forward pass
        raw_scores = weights[input_idx]
        probabilities = softmax(raw_scores)
        
        # Track errors
        loss += -math.log(max(probabilities[target_idx], 1e-15))
        
        # Backward pass (Adjusting the weights)
        for j in range(vocab_size):
            target_value = 1.0 if j == target_idx else 0.0
            gradient = probabilities[j] - target_value
            weights[input_idx][j] -= learning_rate * gradient

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

print("🐊 Training complete! Codegator's matrix is fully calibrated.\n")

# 4. Text Generation Function
def codegator_respond(start_word, num_words=10):
    start_word = start_word.lower().strip()
    if start_word not in word_to_idx:
        # If Codegator doesn't know the word, it picks a random starter
        start_word = random.choice(unique_words)
        
    current_word = start_word
    output = [current_word]
    
    for _ in range(num_words):
        input_idx = word_to_idx[current_word]
        raw_scores = weights[input_idx]
        probabilities = softmax(raw_scores)
        
        # Weighted choice selection
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

# 5. Interactive Loop
print("--- Chat with Codegator ---")
print("Type a single word to seed Codegator's text generation (or type 'exit')\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("🐊 Codegator goes back to sleep. Goodbye!")
            break
        if not user_input:
            continue
            
        # Extract just the first word if the user types a sentence
        first_word = user_input.split()[0]
        
        response = codegator_respond(first_word, num_words=12)
        print(f"Codegator: ... {response} ...\n")
        
    except KeyboardInterrupt:
        print("\n🐊 Codegator out!")
        break

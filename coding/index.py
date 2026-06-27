import os
import random
import math
from pathlib import Path

# ==========================================
# 1. Universal Polyglot File Ingestion Engine
# ==========================================
def harvest_all_active_languages(directory="."):
    """
    Recursively discovers and digests source files across all major active 
    programming paradigms while systematically ignoring discontinued/dead languages.
    """
    
    # Comprehensive whitelist of active, modern, and production-ready language extensions
    ACTIVE_EXTENSIONS = {
        # Systems & Core Infrastructure
        '.c', '.cpp', '.h', '.hpp', '.cc', '.cxx', '.cs', '.java', '.rs', '.go', '.v', '.zig',
        
        # Web Development & Scripting Stacks
        '.js', '.ts', '.jsx', '.tsx', '.py', '.rb', '.php', '.pyw', '.lua', '.dart',
        
        # Mobile, Native, & Cross-Platform
        '.swift', '.kt', '.kts', '.m', '.mm', '.scala', '.groovy', '.pl', '.pm',
        
        # Data Science, Analysis, Query, & Math
        '.sql', '.r', '.jl', '.sas', '.mjs',
        
        # Functional, Logic, & Declarative Programming
        '.hs', '.lhs', '.erl', '.hrl', '.ex', '.exs', '.clj', '.cljs', '.ml', '.mli', '.fs', '.fsi',
        
        # Shells, Scripting, Automation, & Orchestration
        '.sh', '.bash', '.ps1', '.psm1', '.bat', '.cmd', '.awk', '.sed',
        
        # Config, Layout, Styling, & Serialization Pipelines
        '.html', '.htm', '.css', '.scss', '.sass', '.json', '.yaml', '.yml', '.toml', '.xml'
    }
    
    # Strict blacklist of historically discontinued, legacy, or dead programming systems
    DISCONTINUED_EXTENSIONS = {
        '.cob', '.cbl',               # COBOL
        '.f', '.for', '.f90', '.f03',  # Fortran (Legacy formats)
        '.pas', '.pp', '.inc',         # Pascal / Delphi Object Pascal
        '.bas', '.cls', '.frm',        # QuickBASIC / Visual Basic classic (VB6)
        '.alg',                       # ALGOL
        '.pli', '.pl1',                # PL/I
        '.sim',                        # SIMULA
        '.snobol', '.sno',             # SNOBOL
        '.b',                          # B (Precursor to C)
        '.cl', '.lisp',                # Common Lisp / Maclisp (Legacy variants)
        '.adb', '.ads'                 # Ada (Largely sunsetted outside niche defense legacy)
    }

    combined_corpus = []
    print(f"🐊 Scanning '{directory}' for all active development distributions...")
    
    # Iterate through target directory hierarchy
    for path in Path(directory).rglob('*'):
        if path.is_file():
            ext = path.suffix.lower()
            
            # Skip historical, non-active formats immediately
            if ext in DISCONTINUED_EXTENSIONS:
                continue
                
            # Parse only if it's explicitly identified as active
            if ext in ACTIVE_EXTENSIONS:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            combined_corpus.append(content)
                except (IOError, UnicodeDecodeError):
                    # Quietly step past binaries, compilation artifacts, or restricted files
                    continue
                    
    if not combined_corpus:
        print("⚠️ No valid active local source repositories found. Deploying universal polyglot matrix.")
        return """
        def python_gator(x): return [i for i in x]
        const jsGator = (arr) => arr.map(x => x * 2);
        public class JavaGator { public static void main(String[] args) {} }
        fn rust_gator<T>(val: T) -> Option<T> { Some(val) }
        func goGator(ch chan int) { ch <- 42 }
        SELECT gator_id, status FROM production_nodes WHERE active = 1;
        """
    
    print(f"🐊 Successfully compiled {len(combined_corpus)} active codebase file(s) into matrix.")
    return "\n".join(combined_corpus)

# Execute cross-language codebase collection
CORPUS = harvest_all_active_languages(".")

# --- Universal Structural Syntax Splitter ---
# Isolate multi-language structural delimiters, arithmetic tokens, and logic flags
UNIVERSAL_SYMBOLS = [
    "{", "}", "(", ")", "[", "]", "<", ">", 
    "=", "+", "-", "*", "/", "%", "!", "&", "|", "^",
    ":", ";", ",", ".", "?", '"', "'", "->"
]

for symbol in UNIVERSAL_SYMBOLS:
    CORPUS = CORPUS.replace(symbol, f" {symbol} ")

words = CORPUS.split()
unique_words = sorted(list(set(words)))
vocab_size = len(unique_words)

# Initialize bidirectional network lookups
word_to_idx = {word: i for i, word in enumerate(unique_words)}
idx_to_word = {i: word for i, word in enumerate(unique_words)}

print(f"Vocabulary Size: {vocab_size} distinct multi-language elements available.")

# ==========================================
# 2. Network Hyperparameters & Parameters
# ==========================================
# Latent state dimensionality expanded to handle universal multi-paradigm mappings
hidden_size = 64  
learning_rate = 0.05
epochs = 1000

def init_matrix(rows, cols):
    bound = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]

def init_vector(size):
    return [0.0 for _ in range(size)]

# Network Weights and Biases Allocation
W1 = init_matrix(vocab_size, hidden_size)
b1 = init_vector(hidden_size)
W2 = init_matrix(hidden_size, vocab_size)
b2 = init_vector(vocab_size)

# ==========================================
# 3. Network Non-Linearities & Activations
# ==========================================
def tanh(vector):
    return [math.tanh(x) for x in vector]

def softmax(vector):
    max_val = max(vector)
    exp_vector = [math.exp(x - max_val) for x in vector]
    sum_exp = sum(exp_vector)
    return [x / sum_exp for x in exp_vector]

# ==========================================
# 4. Neural Optimization Engine
# ==========================================
print(f"\n🐊 Codegator is optimizing weights over global multi-language arrays...")

for epoch in range(epochs):
    epoch_loss = 0
    
    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i+1]
        
        input_idx = word_to_idx[current_word]
        target_idx = word_to_idx[next_word]
        
        # --- FORWARD STEP ---
        hidden_raw = [W1[input_idx][h] + b1[h] for h in range(hidden_size)]
        hidden_activated = tanh(hidden_raw)
        
        output_raw = [b2[out] for out in range(vocab_size)]
        for out in range(vocab_size):
            for h in range(hidden_size):
                output_raw[out] += hidden_activated[h] * W2[h][out]
                
        probabilities = softmax(output_raw)
        
        # Track Loss
        epoch_loss += -math.log(max(probabilities[target_idx], 1e-15))
        
        # --- BACKPROPAGATION ---
        d_output = [prob for prob in probabilities]
        d_output[target_idx] -= 1.0  # Gradient of Cross-Entropy combined with Softmax
        
        dW2 = [[0.0 for _ in range(vocab_size)] for _ in range(hidden_size)]
        db2 = [grad for grad in d_output]
        
        for h in range(hidden_size):
            for out in range(vocab_size):
                dW2[h][out] = hidden_activated[h] * d_output[out]
                
        d_hidden_activated = [0.0 for _ in range(hidden_size)]
        for h in range(hidden_size):
            for out in range(vocab_size):
                d_hidden_activated[h] += d_output[out] * W2[h][out]
                
        d_hidden_raw = [d_hidden_activated[h] * (1.0 - hidden_activated[h] ** 2) for h in range(hidden_size)]
        
        dW1_row = [grad for grad in d_hidden_raw]
        db1 = [grad for grad in d_hidden_raw]
        
        # --- GRADIENT DESCENT UPDATES ---
        for out in range(vocab_size):
            b2[out] -= learning_rate * db2[out]
            for h in range(hidden_size):
                W2[h][out] -= learning_rate * dW2[h][out]
                
        for h in range(hidden_size):
            b1[h] -= learning_rate * db1[h]
            W1[input_idx][h] -= learning_rate * dW1_row[h]

    if (epoch + 1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:04d}/{epochs} | Cross-Language Syntax Loss: {epoch_loss:.4f}")

print("🐊 Training sequence completed. Weights structural matrix locked.\n")

# ==========================================
# 5. Generative Inference Engine
# ==========================================
def codegator_respond(start_word, num_words=20):
    if start_word not in word_to_idx:
        start_word = random.choice(unique_words)
        
    current_word = start_word
    output = [current_word]
    
    for _ in range(num_words):
        input_idx = word_to_idx[current_word]
        
        hidden_raw = [W1[input_idx][h] + b1[h] for h in range(hidden_size)]
        hidden_activated = tanh(hidden_raw)
        
        output_raw = [b2[out] for out in range(vocab_size)]
        for out in range(vocab_size):
            for h in range(hidden_size):
                output_raw[out] += hidden_activated[h] * W2[h][out]
                
        probabilities = softmax(output_raw)
        
        # Weighted categorical sampling
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
        
    # Reassemble code structural delimiters back to uniform code line formations
    raw_code = " ".join(output)
    for symbol in UNIVERSAL_SYMBOLS:
        raw_code = raw_code.replace(f" {symbol} ", symbol)
    return raw_code

# ==========================================
# 6. Interactive Terminal Context Interface
# ==========================================
print("--- Chat with Universal Active-Polyglot Codegator ---")
print("Provide an active syntax token (e.g., 'fn', 'const', 'func', 'SELECT', 'public', 'def') to begin.\n")

while True:
    try:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("🐊 Codegator goes back to sleep. Goodbye!")
            break
        if not user_input:
            continue
            
        first_word = user_input.split()[0]
        response = codegator_respond(first_word, num_words=20)
        print(f"Codegator:\n{response}\n")
        
    except KeyboardInterrupt:
        print("\n🐊 Codegator out!")
        break

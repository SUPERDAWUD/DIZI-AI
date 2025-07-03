import numpy as np
import string
from nltk.stem.porter import PorterStemmer
import subprocess
import sys
import os
import socket

stemmer = PorterStemmer()

def tokenize(sentence):
    # Lowercase, split by space, and strip punctuation
    return [word.strip(string.punctuation) for word in sentence.lower().split()]

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Stem and match words to one-hot encode into vector
    sentence_words = [stem(w) for w in tokenized_sentence if w not in string.punctuation]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

def start_generator_api():
    # Only start if not already running
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 5050))
        s.close()
        return  # Already running
    except Exception:
        pass
    # Start generator_api.py in a new process
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "generator_api.py")
    subprocess.Popen([python_exe, script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



# src/text_prep.py
import string

class Text_prep():
    def __init__(self, text):
        self.text = text

    def to_lowercase(self):
        self.text = self.text.lower()

    def remove_punctuation(self):
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self):
        return self.text.split()

    def preprocess(self):
        self.to_lowercase()
        self.remove_punctuation()
        return self.tokenize()
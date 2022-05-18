from textEncoder.countVectorizer import countVectorizer
from textEncoder.tfidfVectorizer import tfidfVectorizer
from textEncoder.hashingVectorizer import hashingVectorizer
from activations.binaryStep import binaryStep

text = ["The quick brown fox jumped over the lazy dog."]

print(countVectorizer(text))
print(tfidfVectorizer(text))
print(hashingVectorizer(text))

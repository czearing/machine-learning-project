from sklearn.feature_extraction.text import HashingVectorizer


def hashingVectorizer(incomingValue):
    vectorizer = HashingVectorizer(n_features=20)

    # encode document
    vector = vectorizer.transform(incomingValue)

    return vector.toarray()

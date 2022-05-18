from sklearn.feature_extraction.text import CountVectorizer


def countVectorizer(incomingValue):
    vectorizer = CountVectorizer()

    # tokenize and build vocab
    vectorizer.fit(incomingValue)

    # encode document
    vector = vectorizer.transform(incomingValue)

    return vector.toarray()

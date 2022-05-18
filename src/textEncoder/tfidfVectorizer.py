from sklearn.feature_extraction.text import TfidfVectorizer


def tfidfVectorizer(incomingValue):
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    vectorizer.fit(incomingValue)

    # encode document
    vector = vectorizer.transform([incomingValue[0]])

    return vector.toarray()

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


german_stop_words = stopwords.words('german')
german_stop_words.extend(['000', 'page', 'de','www','the','http','of', 'nen', 'spiegel','sagt', 'sagte', 'sei','gesamtseiten','taz', 'welt','pdf', 'seit','dpa','zeit', 'seien', '2012', '2013', '2015', '2016', '2023'])

def top_tfidf_terms(df,newspaper, years, top_n=5):
    # Global IDF values for the entire corpus
    vectorizer_global = TfidfVectorizer(max_df=0.80, min_df=0.01, stop_words=german_stop_words, use_idf=True)
    _ = vectorizer_global.fit_transform(df['Extracted Text'])  # Fitting to obtain global IDF values only

    # Filter by newspaper if specified, else use the whole DataFrame
    if newspaper != 'ALL':
        filtered_df = df[df['Newspaper'] == newspaper]
    else:
        filtered_df = df

    # Filter by years if specified
    if years != 'ALL':
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]

    # Extracted text from the filtered subset
    filtered_articles = filtered_df['Extracted Text']

    # Local vectorizer without recalculating IDF values
    vectorizer_local = TfidfVectorizer(vocabulary=vectorizer_global.vocabulary_, stop_words=german_stop_words, use_idf=False)
    tf_matrix = vectorizer_local.fit_transform(filtered_articles)

    # Apply global IDF values and calculate TF-IDF for the subset
    tfidf_matrix = tf_matrix.multiply(vectorizer_global.idf_)

    # Determine top terms
    feature_names = vectorizer_global.get_feature_names_out()
    dense_matrix = tfidf_matrix.todense()
    tfidf_means = dense_matrix.mean(axis=0).tolist()[0]
    term_scores = [(feature_names[i], tfidf_means[i]) for i in range(len(feature_names))]
    sorted_term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)

    return sorted_term_scores[:top_n]


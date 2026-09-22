import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle # save model

import string
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def load_rf_data(dataframe, ):

    df = dataframe.copy()
    df.rename(columns={"Review": "text", "Sentiment":"review"}, inplace=True)
    df.drop(['UserID'], axis=1, inplace=True)

    # adjust labels
    df.loc[ df['review'] == 1, 'review'] = 2
    df.loc[ df['review'] == 0, 'review'] = 1
    df.loc[ df['review'] ==-1, 'review'] = 0

    # data cleaning
    df["text_clean"] = df["text"].apply(lambda x: clean_text(x))

    # create doc2vec vector columns
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["text_clean"].apply(lambda x: x.split(" ")))]

    # train a Doc2Vec model with our text data
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    # transform each document into a vector data
    doc2vec_df = df["text_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    df = pd.concat([df, doc2vec_df], axis=1)

    # add number of characters column
    df["nb_chars"] = df["text"].apply(lambda x: len(x))

    # add number of words column
    df["nb_words"] = df["text"].apply(lambda x: len(x.split(" ")))

    # add tf-idfs columns
    tfidf = TfidfVectorizer(min_df = 10)
    tfidf_result = tfidf.fit_transform(df["text_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)

    label = "review"
    ignore_cols = [label, "text", "text_clean"]
    features = [c for c in df.columns if c not in ignore_cols]

    test_df = df[features]
    test_ys = df['review']

    return test_df, test_ys


def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

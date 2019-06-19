from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import json


class PreProcessing:
    """This class contains the methods to preprocess the text loaded from corpus,
    which is later used for building inverted index.
    """
    def __init__(self):
        """
        contains nltk stopwords list and most frequent and unhelpful terms from the text
        """
        # nltk stopwords list
        self.STOPWORDS = list(stopwords.words('english'))
        # most frequent stems in the corpus are considered not helpful to the queries
        self.FREQ_STEMMED = ['film', '2018', 'direct', 'releas', 'star', 'written', 'produc', 'one', 'also', '2016']

    def flatten(self, x):
        """
        leave 1D list unchanged, strings to a list, multi-D list to 1D list
        :param x:
        :return: 1D list
        """
        if isinstance(x, list):
            return [ele for sub in x for ele in self.flatten(sub)]
        else:
            return [x]

    def normalize(self, token):
        """do case-folding, removing stopwords, and stemming."""
        if token.lower() in self.STOPWORDS or len(token) < 2:
            # remove stopwords and 1-char word
            return ' '
        else:
            stemmer = PorterStemmer()
            stemmed_token = stemmer.stem(token.lower())
            if stemmed_token in self.FREQ_STEMMED:
                return ' '
            else:
                return stemmed_token

    def test_corpus(self, filename='test_corpus.json'):
        """Create a test containing 10 hand-made documents corpus in json file."""
        test_corpus = defaultdict(dict)
        titles = ['Dark Knight', 'Harry Potter 3', 'Pan\'s Labyrinth' 'Amores Perros', 'Some Movie', 'Titanic',
                 'Moholland Road', 'Todd', 'Psycho', 'TBD']
        directors = ['Christopher Nolan', 'Alfonso Cuaron', 'Guillermo de Toro', 'Alejandro Gonzales',
                     'Gael Garcia Bernal', 'Steven Spielberg', 'David Fincher', 'Tim Burton',
                     'Alfred Hitchcock', 'Jin Zhao']
        starrings = ['Ledger', 'Daniel Radcliff', 'Some Actor', ['Gael Garcia Penal', 'Some Actress'],
                     ['Anne Hathaway', 'Anna Kendrick', 'Blake Lively'], ['Kate Winslate', 'Leonardo Diccaprio'],
                     'Naomi Watts', ['Helena Carter', 'Johny Depp'], ['Scary', 'Horror'], 'Jin Zhao']
        locations = ['USA', 'us', 'America', 'CA', 'UK', 'uk', 'canada', 'china', 'MA', '']
        texts = ['The look of Dark Knight is so cool!', 'Harry Potter 3 is my favorite Harry Potter movie',
                 'This movie is kind of scary', 'Gael looks good in this movie', 'this is an imaginary movie',
                 'The ship in the movie is also very beautiful', 'the dream descriptions are so uncanny',
                 'I love the songs in it', 'another scary but tasteful film', 'I want to make a movie sometime']
        # movie free text
        for i, title, director, starring, location, text \
                in zip(range(10), titles, directors, starrings, locations, texts):
            test_corpus[str(i)]['Title'] = [title]
            test_corpus[str(i)]['Director'] = [director]
            test_corpus[str(i)]['Starring'] = [starring]
            test_corpus[str(i)]['Location'] = [location]
            test_corpus[str(i)]['Text'] = text

        json_obj = json.dumps(test_corpus, sort_keys=True, indent=2)

        with open(filename, 'w') as f:
            f.write(json_obj)

        print('test corpus has been created!')

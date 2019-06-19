import shelve, json, heapq, time
from nltk import word_tokenize
from math import sqrt, log10
from preprocessing import PreProcessing
from collections import defaultdict, Counter

preprocess = PreProcessing()


def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        time_taken = end - start
        print("it takes {} seconds to build the index".format(time_taken))
    return wrapper


class VSM_index():
    def __init__(self):
        # number of document, used in computing idf
        self.N = None
        # dict containing key_value pairs: {docID : {'term': tf}}
        self.doc_term_counter = dict()

    def tf(self, term_freq):
        return 1 + log10(term_freq) if term_freq > 0 else 0.0

    def idf(self, doc_num, df):
        return log10(doc_num/df)

    @timing
    def inverted_index(self, index_shelve_name, len_normalization_shelve_name, corpus_name='2018_movies.json'):
        """create vsm_index.db with key_value pairs: {'term':[(docID, tf), ...]}
        create term_normalization.db with key_value pairs:{'docID': the length of the vector,...}"""
        with open(corpus_name) as f:
            corpus_json = json.load(f) # <class 'dict'>
        # number of document, used in computing idf
        self.N = len(corpus_json)  #1923

        with shelve.open(index_shelve_name, writeback=False) as vsm_index:
            # traverse every movie_info_dict in each document
            docIDs = sorted(corpus_json.keys(), key=lambda d: int(d))
            for docID in docIDs:
                # # HASH: Out of overflow pages.  Increase page size
                # choosing 800 hundreds because it seems that one shelve file cannot hold more entries
                # and the postings lists got cut off in the middle
                if int(docID) > 800:
                    break
                # get the tokens from title and text
                title = corpus_json[docID]['Title'] # <class 'list'>
                text = corpus_json[docID]['Text'] # <class 'str'>
                tokens = word_tokenize(str(title) + ' ' + text)
                # normalize those tokens
                terms = [preprocess.normalize(token) for token in tokens if preprocess.normalize(token)]
                # create a dict with key_value pairs: {'term': tf}
                term_counter = Counter()
                term_counter.update(terms)
                # dict containing key_value pairs: {docID : {'term': tf}}
                self.doc_term_counter[docID] = term_counter

                # fill vsm_index.db with key_value pairs: {'term':[(docID, df), ...]}
                for term in term_counter.keys():
                    if term in vsm_index.keys():
                        lst = vsm_index[term]
                        lst.append((docID, term_counter[term]))
                        vsm_index[term] = lst
                    else:
                        vsm_index[term] = [(docID, term_counter[term])]

                # append the df to the last element of the postings
            for term in vsm_index.keys():
                df = len(vsm_index[term])
                new_posting = vsm_index[term]
                new_posting.append(df)
                vsm_index[term] = new_posting
            print("vsm_index.db has been created!")

            # fill len_normalization.db with key_value pairs: 'docID': the length of the vector
            with shelve.open(len_normalization_shelve_name, writeback=False) as len_normalization:
                # traverse doc_term_counter
                for docID in docIDs:
                    if int(docID) > 800:
                        break
                    sum_tfidf_2pow = 0.0
                    for term, freq in self.doc_term_counter[docID].items():
                        sum_tfidf_2pow += pow(self.tf(freq) * self.idf(self.N, vsm_index[term][-1]), 2)
                    # this vector_length is used for normalization
                    vector_length = sqrt(sum_tfidf_2pow)
                    len_normalization[docID] = vector_length
            print("len_normalization.db has been created!")

    @timing
    def corpus_shelve(self, shelvename, corpus_name='2018_movies.json'):
        """store the info from the corpus json file to corpus shelve file for easy access"""
        with open(corpus_name, 'r') as f:
            corpus = json.load(f)

        with shelve.open(shelvename, writeback=False) as corpus_shelve:
            for docID, movie_info_dict in corpus.items():
                starring_list = preprocess.flatten(movie_info_dict['Starring'])

                corpus_shelve[docID] = {'Title': movie_info_dict['Title'],
                                        'Director': movie_info_dict['Director'],
                                        'Location': movie_info_dict['Location'],
                                        'Starring': ', '.join(starring_list),
                                        'Text': movie_info_dict['Text']}
        print('corpus shelve has been created!')

    def cosine_score_disjunct(self, query, k, inverted_index='vsm_index', doc_normalize ='len_normalization'):
        """
        :param query: query term
        :param k: top k results that the user wants
        :param inverted_index: vsm_index.db
        :param doc_normalize: len_normalization.db
        :return: a list of tuples (doc's cosine_similarity score, docID), a list of stop words, and a list of unknown words
        compute the cosine scores of a disjunctive query for each document and return top k documents
        """
        score_queue = []
        stopwords = []
        unknown = []
        query_term_list = []
        query_list = query.split(' ')

        with shelve.open(inverted_index, writeback=False) as vsm_index:
            for token in query_list:
                # normalize the input query and get lists of valid terms, unknown and stopwords
                normalized = preprocess.normalize(token)
                if normalized:
                    if normalized is ' ':
                        stopwords.append(str(token))
                    else:
                        if normalized not in vsm_index.keys():
                            unknown.append(str(token))
                        else:
                            query_term_list.append(normalized)
            # store the cosine similarity score of every document before normalization
            scores = defaultdict(float)

            with shelve.open(doc_normalize, writeback=False) as len_norm:
                self.N = len(len_norm)
                # stores key_value pairs in query: {'term': freq}
                query_term_counter = Counter(query_term_list)

                # get dot product of vector q and vector d
                for term in query_term_list:
                    posting = vsm_index[term]
                    idf = self.idf(self.N, vsm_index[term][-1]) #  query term idf and doc term idf are the same
                    query_tfidf = self.tf(query_term_counter[term]) * idf
                    for docID, freq in posting[:-1]:
                        doc_tfidf = self.tf(freq) * idf
                        scores[str(docID)] += query_tfidf * doc_tfidf  # before normalized

                for docID in scores:
                    # use priority queue to select top k documents
                    scores[docID] /= len_norm[docID]  # normalization
                    if len(score_queue) < k:
                        heapq.heappush(score_queue, (scores[docID], docID))
                        heapq.heapify(score_queue)
                    else:
                        # this is more efficient push and then pop
                        heapq.heappushpop(score_queue, (scores[docID], docID))

        return sorted(score_queue, reverse=True), stopwords, unknown

if __name__ == '__main__':
    # create test_corpus
    # preprocess.test_corpus()
    vsm_index = VSM_index()
    vsm_index.inverted_index(index_shelve_name='vsm_index', len_normalization_shelve_name='len_normalization')
    vsm_index.corpus_shelve(shelvename='corpus_shelve')







from flask import Flask, render_template, request
from vsm_index import VSM_index
from nltk import sent_tokenize
import shelve


# this is a helper function
def dummy_movie_data(docID, shelvename='corpus_shelve'):
    """Return data fields for a movie."""
    with shelve.open(shelvename, writeback=False) as corpus_shelve:
        movie_object = corpus_shelve[docID]
    return movie_object

# this is a helper function
def dummy_movie_snippet(scores_pair):
    """
    :param  score_queue returned from cosine_score_disjunct
    Return a snippet for the results page.
    """
    score = scores_pair[0]
    docID = scores_pair[1]

    movie_object = dummy_movie_data(docID, shelvename='corpus_shelve')
    # display the title
    title = movie_object['Title']
    # display the first two sentences of the free text for each movie on the results page
    description = ' '.join(sent_tokenize(movie_object['Text'])[:2])
    result = (docID, title, description, score)
    return result


# Create an instance of the flask application within the appropriate namespace (__name__).
# By default, the application will be listening for requests on port 5000.
app = Flask(__name__)


# Welcome page
# Python decorators are used by flask to associate url routes to functions.
@app.route("/")
def query():
    """For top level route ("/"), simply present a query page."""
    return render_template('query_page.html')


# This takes queries and turns them into results
@app.route("/results/<int:page_num>", methods=['POST'])
def results(page_num):
    """Generate a result set for a query and present the 10 results starting with <page_num>."""
    vsm_index = VSM_index()
    raw_query = request.form['query']
    scores, skippedwords, unk = vsm_index.cosine_score_disjunct(raw_query, k=30)

    if unk:
        return render_template('error_page.html', unknown_terms=unk)
    else:
        # render the results page
        num_hits = len(scores)  # Save the number of hits to display later
        scores_pair = scores[((page_num - 1) * 10):(page_num * 10)]  # Limit of 10 results per page
        movie_results = list(map(dummy_movie_snippet, scores_pair))  # Get movie snippets: title, abstract, etc.
        return render_template('results_page.html', orig_query1=raw_query,
                               results=movie_results, srpn=page_num, len=len(scores_pair), skipped_words=skippedwords, total_hits=num_hits)


# Process requests for movie_data pages
@app.route('/movie_data/<film_id>')
def movie_data(film_id):
    """Given the doc_id for a film, present the title and text (optionally structured fields as well)
    for the movie."""
    data = dummy_movie_data(film_id, shelvename='corpus_shelve')  # Get all of the info for a single movie
    return render_template('doc_data_page.html', data=data)

@app.route("/results/more/<int:page_num>", methods=['POST'])
def more_data(page_num):
    """use the selected current document's title and text as a query to search for similar documents"""
    vsm_index = VSM_index()
    raw_query = request.form['more']
    scores, skippedwords, unk = vsm_index.cosine_score_disjunct(query=raw_query, k=20)
    # render the results page
    num_hits = len(scores)  # Save the number of hits to display later
    scores_pair = scores[((page_num - 1) * 10):(page_num * 10)]  # Limit of 10 results per page
    movie_results = list(map(dummy_movie_snippet, scores_pair))  # Get movie snippets: title, abstract, etc.
    return render_template('more_like_this.html', orig_query=raw_query,
                            results=movie_results, srpn=page_num, len=len(scores_pair), total_hits=num_hits)


# If this module is called in the main namespace, invoke app.run()
if __name__ == "__main__":
    app.run()

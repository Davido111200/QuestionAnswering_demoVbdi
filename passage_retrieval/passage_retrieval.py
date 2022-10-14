from .bm25 import BM25Okapi, BM25L
from .dpr import DPR
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import re

from .googlesearch import GoogleSearch

class PassageRetrieval:
    """
    Passages retrieval
    """
    def __init__(self, search_engine):
        """ Init Passage retrieval

        Args:
            search_engine (str): Search engine name
        """
        self.name = search_engine
        if search_engine == 'bm25':
            self.search_engine = BM25Okapi
        elif search_engine == 'bm25l':
            self.search_engine = BM25L
        elif search_engine == 'dpr':
            self.search_engine = DPR()

    def split_passages(self, html_page, num_sent=3):
        """Split html page into passages

        Args:
            html_page (string): string from html page
            num_passage (int, optional): number of sentences per passage. Defaults to 10.
        """
        setences = sent_tokenize(html_page)
        passages = []
        for i in range(0, len(setences), num_sent):
            passages.append('. '.join(setences[i:i+num_sent]))

        return passages

    def get_score_one_page(self, query, list_passages):
        """_summary_

        Args:
            query (str): Query
            list_passages (list): List of passages from one page

        Returns:
            numpy.Array: Array of score
        """
        scores = self.search_engine.get_scores(query, list_passages)
        return scores

    def get_top_k_one_page(self, query, list_passages, k=5):
        """Get top k passages from one page

        Args:
            query (str): Query
            list_passages (list): List of passages from one page
            k (int, optional): Number of passage. Defaults to 10.

        Returns:
            _type_: _description_
        """
        if self.name != 'dpr':
            scores = self.search_engine(corpus=list_passages).get_scores(query)
        else:
            scores = self.search_engine.get_scores(query, list_passages)
        top_k = np.argsort(scores)[::-1][:k]
        res = []
        for i in top_k:
            res.append((scores[i],list_passages[i]))
        return res

    def search(self, query, list_pages, k=10):
        """Search query in list of pages

        Args:
            query (string): query
            list_pages (list): list of pages
            k (int, optional): number of top passages. Defaults to 10.

        Returns:
            list: list of top passages
        """
        top_passages = []
        for page in list_pages:
            passages = self.split_passages(page)
            if len(passages) < 3:
                continue
            top_passages.extend(self.get_top_k_one_page(query, passages, k))

        top_passages.sort(key=lambda x: x[0], reverse=True)
        return top_passages[:k]


# remove none-ascii characters by regural expression
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

if __name__ == "__main__":
    # test
    gg = GoogleSearch()
    #start_time = time.time()
    print("Start")
    res = gg.search("Where karl marx was born?", 10)
    search = PassageRetrieval('dpr')
    result = search.search("Where karl marx was born?", res, 10)
    print()

    for i in range(len(result)):
        print(result[i][1])
        print()
    #print(result)
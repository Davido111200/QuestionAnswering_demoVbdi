from ast import If
from multiprocessing import pool
from operator import le
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re
import time
from ftfy import fix_encoding
from langdetect import detect


class GoogleSearch:
    """Google Search with API
    """
    def __init__(self, gg_api="AIzaSyCe6HDQsgJ3Nw42G0KZiGH4VL6afluQg-A", gg_cse='e42ceac5c43b04b50') -> None:
        self.google_api = gg_api
        self.google_cse = gg_cse
        self.pool = pool.Pool(10)
        self.service = build("customsearch", "v1", developerKey=self.google_api)

    def search(self, query, start=0, num=5):
        """Search query in google and return list of top k url

        Args:
            query (str): Query string
            start (int, optional): _description_. Defaults to 1.
            num (int, optional): Number of page. Defaults to 10.

        Returns:
            str: List of result of google searching
        """
        try:
            res = self.service.cse().list(
                q=query,
                cx=self.google_cse,
                start=start,
                num=num,
            ).execute()
            list_content = self.pool.map(self.get_content, [item['link'] for item in res['items']])
            
            return list_content
        except:
            return []

    def get_content(self, url):
        """Get content from url

        Args:
            url (str): an url

        Returns:
            str: Text information from url
        """
        
        try: 
            print(url)
            response = requests.get(url, timeout=0.5)
            soup = BeautifulSoup(response.text, 'html.parser')
            #soup = soup.get_text()
            soup = ' '.join([p.text for p in soup.find_all('p')])
            soup = fix_encoding(soup)
            soup = re.sub(' +', ' ', soup)
            soup = re.sub('\n+', '\n', soup)
            soup = re.sub('\t+', '\t', soup)
            soup = re.sub(r'[^\x00-\x7F]+', ' ', soup) if detect(soup) == 'en' else soup

            return soup
        except:
            return 'None'

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == "__main__":
    gg = GoogleSearch()
    start_time = time.time()
    print("Start")
    res = gg.search("Ai là người giàu nhất việt nam?")
    print("Time: ", time.time() - start_time)
    print("Stop")
    print()
    print()
    #

    #k = res[0].split('.')

    #print(k[1])

    
from multiprocessing import pool
from operator import le
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import re
import time


class GoogleSearch:
    """
    """
    def __init__(self, gg_api="AIzaSyCe6HDQsgJ3Nw42G0KZiGH4VL6afluQg-A", gg_cse='e42ceac5c43b04b50') -> None:
        self.google_api = gg_api
        self.google_cse = gg_cse
        self.pool = pool.Pool(10)
        self.service = build("customsearch", "v1", developerKey=self.google_api)

    def search(self, query, start=1, num=10):
        """
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
        """
        """
        try:
            print(url)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()
        except:
            return ''

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
    res = gg.search("Hà Nội là thủ đô của Việt Nam không?", 10)
    print("Time: ", time.time() - start_time)
    print("Stop")
    print()
    print()

    if len(res)!=0:
        print(res[0])

    
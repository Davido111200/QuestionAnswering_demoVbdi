import math
import numpy as np
from multiprocessing import Pool

class BM25:
    """
    BM25 abstract class for passage retrieval
    """
    def __init__(self, corpus, tokinizer=None) -> None:
        self.tokenizer = tokinizer
        self.corpus_len = 0
        self.avgdl = 0
        self.df = []
        self.idf = {}
        self.doc_len = []

        corpus = self.tokenize(corpus)
        nd = self.initialize(corpus)
        self.initialize_idf(nd)

        self.pool = Pool(4)

    def initialize(self, corpus):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.df.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_len += 1

        self.avgdl = num_doc / self.corpus_len
        return nd

    def initialize_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    """def get_batch_scores(self, queries, doc_ids):
        raise NotImplementedError()"""

    def get_top_k(self, query, docs,k=10):
        scores = self.get_scores(query)
        top_k = np.argsort(scores)[::-1][:k]
        res = []
        for i in top_k:
            res.append((docs[i], scores[i]))
        return res

    def tokenize(self, corpus):
        pool = Pool(cpu_count())
        return pool.map(self.tokenizer, corpus)


class BM25Okapi(BM25):
    """
    BM25 implementation for passage retrieval
    """
    def __init__(self, corpus, tokinizer=None, k1=1.5, b=0.75, ep=0.25) -> None:
        self.k1 = k1
        self.b = b
        self.ep = ep
        super().__init__(corpus, tokinizer)

    def initialize_idf(self, nd):
        negative_idfs = []
        for word, freq in nd.items():
            self.idf[word] = math.log((self.corpus_len - freq + 0.5) / (freq + 0.5))
            if self.idf[word] < 0:
                negative_idfs.append(word)

        self.avg_idf = sum(map(lambda k: float(self.idf[k]), self.idf.keys())) / len(self.idf.keys())
        eps = self.ep*self.avg_idf

        for word in self.idf.keys():
            self.idf[word] += eps

        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        score = [0.0 for _ in range(self.corpus_len)]
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.df])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_top_k(self, query, docs, k=10):
        return super().get_top_k(query, docs, k)

    """def get_batch_scores(self, queries, doc_ids):
        return super().get_batch_scores(queries, doc_ids)"""

class BM25L(BM25):
    """
    BM25L implementation for passage retrieval
    """
    def __init__(self, corpus, tokinizer=None, k1=1.5, b=0.75, delta=0.5) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokinizer)
        #self.pool = Pool(cpu_count())
    
    def initialize_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log((self.corpus_len - freq + 0.5) / (freq + 0.5))
    
    def get_scores(self, query):
        score = [0.0 for _ in range(self.corpus_len)]
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.df])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

if __name__ == "__main__":
    corpus = [
        "Karl Heinrich Marx FRSA (German: [maʁks]; 5 May 1818 – 14 March 1883) was a German philosopher, economist, historian, sociologist, political theorist, journalist, critic of political economy, and socialist revolutionary. His best-known titles are the 1848 pamphlet The Communist Manifesto and the four-volume Das Kapital (1867–1883). Marx's political and philosophical thought had enormous influence on subsequent intellectual, economic, and political history. His name has been used as an adjective, a noun, and a school of social theory.".lower(),
        "Born in Trier, Germany, Marx studied law and philosophy at the universities of Bonn and Berlin. He married German theatre critic and political activist Jenny von Westphalen in 1843. Due to his political publications, Marx became stateless and lived in exile with his wife and children in London for decades, where he continued to develop his thought in collaboration with German philosopher Friedrich Engels and publish his writings, researching in the British Museum Reading Room.".lower(),
        "Marx's critical theories about society, economics, and politics, collectively understood as Marxism, hold that human societies develop through class conflict. In the capitalist mode of production, this manifests itself in the conflict between the ruling classes (known as the bourgeoisie) that control the means of production and the working classes (known as the proletariat) that enable these means by selling their labour-power in return for wages.".lower(),
        "Employing a critical approach known as historical materialism, Marx predicted that capitalism produced internal tensions like previous socioeconomic systems and that those would lead to its self-destruction and replacement by a new system known as the socialist mode of production. For Marx, class antagonisms under capitalism—owing in part to its instability and crisis-prone nature—would eventuate the working class's development of class consciousness, leading to their conquest of political power and eventually the establishment of a classless, communist society constituted by a free association of producers".lower(),
        "Marx actively pressed for its implementation, arguing that the working class should carry out organised proletarian revolutionary action to topple capitalism and bring about socio-economic emancipation.[5]".lower(), 
        "Marx has been described as one of the most influential figures in human history, and his work has been both lauded and criticised.[6] His work in economics laid the basis for some current theories about labour and its relation to capital.[7][8][9] Many intellectuals, labour unions, artists, and political parties worldwide have been influenced by Marx's work, with many modifying or adapting his ideas. Marx is typically cited as one of the principal architects of modern social science.".lower()
    ]

    def tokenizer(text):
        return text.split()
    #tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(corpus, tokinizer=tokenizer)


    query = "Karl Marx birthday".lower()
    tokenized_query = tokenizer(query)

    a = bm25.get_top_k(tokenized_query, corpus, k=3)
    print(query)
    print()
    for i in range(len(a)):
        print(a[i])
        print()

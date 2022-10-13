from transformers import DPRReader, DPRReaderTokenizer
import numpy as np

class DPR:

    def __init__(self, device='cuda') -> None:
        self.device = device
        self.tokenizer = DPRReaderTokenizer.from_pretrained('facebook/dpr-reader-single-nq-base')#.to('cuda')
        self.reader = DPRReader.from_pretrained('facebook/dpr-reader-single-nq-base').to(self.device)
        
    def get_scores(self, query, docs):
        if not isinstance(query, list):
            query = [query]

        if not isinstance(docs, list):
            docs = [docs]

        query = query * len(docs)
        encoded_inputs = self.tokenizer(questions=query,texts=docs, return_tensors='pt',padding=True, truncation=True).to(self.device)
        scores = self.reader(**encoded_inputs).relevance_logits
        scores = scores.detach().cpu().numpy()
        return softmax(scores)


    def get_top_k(self, query, docs, k=2):
        scores = self.get_scores(query, docs)
        top_k = np.argsort(scores)[::-1][:k]
        res = []
        for i in top_k:
            res.append((docs[i], scores[i]))
        return res


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
    dpr = DPR()
    query = ["where marx was born?"]
    docs = [
        "Karl Heinrich Marx FRSA (German: [maʁks]; 5 May 1818 – 14 March 1883) was a German philosopher, economist, historian, sociologist, political theorist, journalist, critic of political economy, and socialist revolutionary. His best-known titles are the 1848 pamphlet The Communist Manifesto and the four-volume Das Kapital (1867–1883). Marx's political and philosophical thought had enormous influence on subsequent intellectual, economic, and political history. His name has been used as an adjective, a noun, and a school of social theory.".lower(),
        "Born in Trier, Germany, Marx studied law and philosophy at the universities of Bonn and Berlin. He married German theatre critic and political activist Jenny von Westphalen in 1843. Due to his political publications, Marx became stateless and lived in exile with his wife and children in London for decades, where he continued to develop his thought in collaboration with German philosopher Friedrich Engels and publish his writings, researching in the British Museum Reading Room.".lower(),
        "Marx's critical theories about society, economics, and politics, collectively understood as Marxism, hold that human societies develop through class conflict. In the capitalist mode of production, this manifests itself in the conflict between the ruling classes (known as the bourgeoisie) that control the means of production and the working classes (known as the proletariat) that enable these means by selling their labour-power in return for wages.".lower(),
        "Employing a critical approach known as historical materialism, Marx predicted that capitalism produced internal tensions like previous socioeconomic systems and that those would lead to its self-destruction and replacement by a new system known as the socialist mode of production. For Marx, class antagonisms under capitalism—owing in part to its instability and crisis-prone nature—would eventuate the working class's development of class consciousness, leading to their conquest of political power and eventually the establishment of a classless, communist society constituted by a free association of producers".lower(),
        "Marx actively pressed for its implementation, arguing that the working class should carry out organised proletarian revolutionary action to topple capitalism and bring about socio-economic emancipation.[5]".lower(), 
        "Marx has been described as one of the most influential figures in human history, and his work has been both lauded and criticised.[6] His work in economics laid the basis for some current theories about labour and its relation to capital.[7][8][9] Many intellectuals, labour unions, artists, and political parties worldwide have been influenced by Marx's work, with many modifying or adapting his ideas. Marx is typically cited as one of the principal architects of modern social science.".lower(),
        "Karl Heinrich Marx FRSA (German: [maʁks]; 5 May 1818 – 14 March 1883) was a German philosopher, economist, historian, sociologist, political theorist, journalist, critic of political economy, and socialist revolutionary. His best-known titles are the 1848 pamphlet The Communist Manifesto and the four-volume Das Kapital (1867–1883). Marx's political and philosophical thought had enormous influence on subsequent intellectual, economic, and political history. His name has been used as an adjective, a noun, and a school of social theory.".lower(),
        "Born in Trier, Germany, Marx studied law and philosophy at the universities of Bonn and Berlin. He married German theatre critic and political activist Jenny von Westphalen in 1843. Due to his political publications, Marx became stateless and lived in exile with his wife and children in London for decades, where he continued to develop his thought in collaboration with German philosopher Friedrich Engels and publish his writings, researching in the British Museum Reading Room.".lower(),
        "Marx's critical theories about society, economics, and politics, collectively understood as Marxism, hold that human societies develop through class conflict. In the capitalist mode of production, this manifests itself in the conflict between the ruling classes (known as the bourgeoisie) that control the means of production and the working classes (known as the proletariat) that enable these means by selling their labour-power in return for wages.".lower(),
        "Employing a critical approach known as historical materialism, Marx predicted that capitalism produced internal tensions like previous socioeconomic systems and that those would lead to its self-destruction and replacement by a new system known as the socialist mode of production. For Marx, class antagonisms under capitalism—owing in part to its instability and crisis-prone nature—would eventuate the working class's development of class consciousness, leading to their conquest of political power and eventually the establishment of a classless, communist society constituted by a free association of producers".lower(),
        "Marx actively pressed for its implementation, arguing that the working class should carry out organised proletarian revolutionary action to topple capitalism and bring about socio-economic emancipation.[5]".lower(), 
        "Marx has been described as one of the most influential figures in human history, and his work has been both lauded and criticised.[6] His work in economics laid the basis for some current theories about labour and its relation to capital.[7][8][9] Many intellectuals, labour unions, artists, and political parties worldwide have been influenced by Marx's work, with many modifying or adapting his ideas. Marx is typically cited as one of the principal architects of modern social science.".lower()
    ]

    print(dpr.get_top_k(query, docs))
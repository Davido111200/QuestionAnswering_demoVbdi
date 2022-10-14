import sys
  
# appending a path
sys.path.append(r'..\..\AnswerExtraction_demoVbdi')
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,QuestionAnsweringPipeline
import operator
from passage_retrieval.passage_retrieval import PassageRetrieval
from passage_retrieval.googlesearch import GoogleSearch

class AnswerExtractor:
    def __init__(self, model="Palak/microsoft_deberta-large_squad"):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        result = {}
        result['answer'] = [item['answer'] for item in answers]
        result['score'] = [item['score'] for item in answers]
        return result

if __name__ == "__main__":
    question = "Who is Fidel Castro"
    gg = GoogleSearch()
    retriever = PassageRetrieval('dpr')
    passages_ = retriever.search(question , gg.search(question))
    passages = [passages_[i][0] for i in range(len(passages_))]
    answer_extractor = AnswerExtractor()
    answers = answer_extractor.extract(question, passages)
    print(answers)

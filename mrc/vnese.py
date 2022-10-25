from transformers import AutoTokenizer, AutoModelForQuestionAnswering, QuestionAnsweringPipeline
import operator

class AnswerExtractor_vn:
    def __init__(self, model_path="ancs21/xlm-roberta-large-vi-qa"):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
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
from transformers import AutoTokenizer,AutoModelForQuestionAnswering,QuestionAnsweringPipeline
import operator

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
  passages = ["Fidel Castro is Cuba President","Fidel Alejandro Castro Ruz (/ˈkæstroʊ/;[1] American Spanish: [fiˈðel aleˈxandɾo ˈkastɾo ˈrus]; 13 August 1926 – 25 November 2016) was a Cuban revolutionary and politician who was the leader of Cuba from 1959 to 2008, serving as the prime minister of Cuba from 1959 to 1976 and president from 1976 to 2008. Ideologically a Marxist–Leninist and Cuban nationalist, he also served as the first secretary of the Communist Party of Cuba from 1961 until 2011. Under his administration, Cuba became a one-party communist state; industry and business were nationalized, and state socialist reforms were implemented throughout society."]

  answer_extractor = AnswerExtractor()
  answers = answer_extractor.extract(question, passages)
  print(answers)

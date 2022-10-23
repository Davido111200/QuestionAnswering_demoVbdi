# import libraries
import string
from nltk.corpus import stopwords
from textblob import Word


class QuestionProcessing:
    def __init__(self, question=''):
        self.question = question

    # lower case
    def lower_case(self, question):
        question = question.lower()
        return question
    
    # remove punctuation
    def remove_punctuation(self, question):
        question = question.translate(str.maketrans('', '', string.punctuation))
        return question

    # remove extra spaces
    def remove_extra_spaces(self, question):
        question = ' '.join(question.split())
        return question

    # lemmatization 
    def lemmatization(self, question):
        question = ' '.join([Word(word).lemmatize() for word in question.split()])
        return question


    # check sentence spelling
    def check_word_spelling(self, word):
        result = Word(word).spellcheck()
        if result[0][1] > 0.8:
            return result[0][0]
        else:
            return word

    def check_sentence_spelling(self, question):
        res = ' '.join([self.check_word_spelling(word) for word in question.split()])
        return res

    # process question
    def process_question(self, question):
        question = self.lower_case(question)
        question = self.check_sentence_spelling(question)
        question = self.remove_punctuation(question)
        question = self.remove_extra_spaces(question)
        question = self.lemmatization(question)
        return question

if __name__ == '__main__':
    question = 'Whhoo   is  theo    presidexnt ovf  USA?'
    question_processing = QuestionProcessing()
    processed_question = question_processing.process_question(question)
    print(processed_question)
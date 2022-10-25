import streamlit as st
import sys
sys.path.append(r'..\..\AnswerExtraction_demoVbdi')
from passage_retrieval.googlesearch import GoogleSearch
from passage_retrieval.passage_retrieval import PassageRetrieval
from passage_retrieval.language_detection import Detect
import operator
  

from mrc.AnswerExtractionScoring import AnswerExtractor
from mrc.vnese import AnswerExtractor_vn

st.set_page_config(layout="wide", page_icon="🖱️", page_title="Search engine tiny")
st.title('Question Answering')

gg = GoogleSearch()
detector = Detect()

def get_answer(question, retriever_model, answer_extractor):
    passages_ = retriever_model.search(question, gg.search(question))
    passages = [passages_[i][1] for i in range(len(passages_))]
    answers = answer_extractor.extract(question, passages)
    return answers

def get_k_top_answer(answers, k):
    return answers.get("answer")[0:k], answers.get("score")[0:k]


def app():
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    question = st.text_input(
        "Please input your question:",
        "who is the richest woman in the world?",
        key="placeholder",
    )


    if detector.det(str(question)) == 1:
      answer_extractor = AnswerExtractor()
      col1, col2 = st.columns(2)

      with col1:
          type_of_search = st.radio(
              "What type of search algorithm do you want to choose?",
              ('bm25', 'dpr'))

          if type_of_search == 'bm25':
              st.write('You selected bm25. This method takes less time but provides less accurate answers')
          elif type_of_search == 'dpr':
              st.write('You selected dpr. This method takes more time but more accurate answers')
          
          num_answers = st.number_input(
              "How many answers do you want to receive?",
              min_value=1,
              max_value=5,
              format="%d"
          )
          
          done = st.button(
              "Search", 
              key="done_"
          )

      with col2:
          if done:
              with st.spinner(text="🤖 Finding Answers..."):
                  retriever = PassageRetrieval(str(type_of_search))
                  answers = get_answer(question, retriever_model=retriever, answer_extractor=answer_extractor)
                  first_k_answers, first_k_scores = get_k_top_answer(answers, num_answers)
              for idx, answer in enumerate(first_k_answers):
                  st.write(answer, "- Score:", first_k_scores[idx])
              st.balloons()
    
    else:
      answer_extractor = AnswerExtractor_vn()
      col1, col2 = st.columns(2)

      with col1:
          type_of_search = 'bm25'

          num_answers = st.number_input(
              "Bạn muốn nhận bao nhiêu câu trả lời?",
              min_value=1,
              max_value=5,
              format="%d"
          )

          done = st.button(
              "Tìm kiếm", 
              key="done_"
          )

      with col2:
          if done:
              with st.spinner(text="🤖 Đang tìm kiếm"):
                  retriever = PassageRetrieval(str(type_of_search))
                  answers = get_answer(question, retriever_model=retriever, answer_extractor=answer_extractor)
                  first_k_answers, first_k_scores = get_k_top_answer(answers, num_answers)
              for idx, answer in enumerate(first_k_answers):
                  st.write(answer, "- Score:", first_k_scores[idx])
              st.balloons()
    

if __name__ == "__main__":
    app()


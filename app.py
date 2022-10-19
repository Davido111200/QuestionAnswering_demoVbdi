import streamlit as st
import sys
sys.path.append(r'..\..\AnswerExtraction_demoVbdi')
from passage_retrieval.googlesearch import GoogleSearch
from passage_retrieval.passage_retrieval import PassageRetrieval
import operator
  

from mrc.AnswerExtractionScoring import AnswerExtractor

st.set_page_config(layout="wide", page_icon="🖱️", page_title="Search engine tiny")
st.title('Question Answering')

gg = GoogleSearch()
answer_extractor = AnswerExtractor()

def get_answer(question, retriever_model):
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

    col1, col2 = st.columns(2)

    with col1:
        type_of_search = st.radio(
            "What type of search algorithm do you want to choose?",
            ('bm25', 'dpr'))

        if type_of_search == 'bm25':
            st.write('You selected bm25.')
        elif type_of_search == 'dpr':
            st.write('You selected dpr')
        
        num_answers = st.number_input(
            "How many answers do you want to receive?",
            min_value=1,
            max_value=5,
            format="%d"
        )

        question = st.text_input(
            "Please input your question:",
            "E.g: Who is Fidel Castro?",
            key="placeholder",
        )
        
        done = st.button(
            "Search"
        )


    if done:

        with col2:
            with st.spinner(text="🤖 Finding Answers..."):
                retriever = PassageRetrieval(str(type_of_search))
                answers = get_answer(question, retriever_model=retriever)
                first_k_answers, first_k_scores = get_k_top_answer(answers, num_answers)
            for answer in first_k_answers:
                st.write(answer)
            st.balloons()


if __name__ == "__main__":
    app()


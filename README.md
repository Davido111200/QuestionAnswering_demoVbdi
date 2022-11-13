# AnswerExtraction_demoVbdi

This project aims to extract answer for English questions

## Pipeline

Follow these steps:

- Document Retrieval (GoogleSearch)
- Passage Retiieval using Passage Ranker - Anserini Retrieval
- (Minor) Language Detection - Currently our demo only recognize Vietnamese and English questions
- Machine reading comprehension 

Weight of xlmRoberta: https://drive.google.com/file/d/11VGMWNRjaKLL-SEDrPUTLilehHXG-e_y/view?usp=sharing

## Instructions

1. Open up Streamlit python file
2. Search question in either Vietnamese or English
3. Set up configurations for the type of answer you want to receive
4. DONE!!

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, pipeline
import streamlit as st

summarization_tokenizer = AutoTokenizer.from_pretrained("Blaise-g/longt5_tglobal_large_sumpubmed")

summarization_model = AutoModelForSeq2SeqLM.from_pretrained("Blaise-g/longt5_tglobal_large_sumpubmed") 

qa_tokenizer = AutoTokenizer.from_pretrained("sultan/BioM-ELECTRA-Large-SQuAD2-BioASQ8B")

qa_model = AutoModelForQuestionAnswering.from_pretrained("sultan/BioM-ELECTRA-Large-SQuAD2-BioASQ8B")

question_answer = pipeline("question-answering", model=qa_model, tokenizer = qa_tokenizer)

@st.cache

def summarize(note):
    input_ids = summarization_tokenizer.encode("summarize: " + note, return_tensors="pt", add_special_tokens=True)
    generated_ids = summarization_model .generate(input_ids=input_ids,num_beams=5,max_length=15000,repetition_penalty=2.5,length_penalty=0.75,early_stopping=True,num_return_sequences=3)
    preds = [summarization_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0]

def qa(note, question): 
    answer = question_answer(question = question , context = note)
    return answer['answer']

st.title('Machine Learning Based Clinical Summarization')
st.header('Enter Sample Medical Note:')
note = st.text_area("Enter some text 👇")

if st.button('Generate Summary'):
    summary = summarize(note)
    st.success(summary)


st.header('Ask a question related to the note:')
question = st.text_area("Enter Question ❓")  
if st.button('Generate Answer'):
    answer = qa(note, question)
    st.success(answer)



import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_answer(context, question):
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    QA_input = {'question': question, 'context': context}
    res = nlp(QA_input)
    answer = res['answer']
    return answer

def main():
    st.title("Question Answering App")
    st.markdown("Enter the context and question, then click on 'Get Answer' to retrieve the answer.")

    
    context = st.text_area("Context", "Enter the context here...")
    question = st.text_input("Question", "Enter the question here...")

    
    if st.button("Get Answer"):
        
        if context.strip() == "" or question.strip() == "":
            st.warning("Please enter the context and question.")
        else:
            
            answer = get_answer(context, question)
            st.success(f"Answer: {answer}")


if __name__ == "__main__":
    main()

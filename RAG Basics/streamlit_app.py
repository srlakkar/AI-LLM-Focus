import streamlit as st
from rag_chat import retrieve_and_answer

st.title("PDF RAG Chatbot")
user_query = st.text_input("Ask your question about the PDFs:")

if st.button("Get Answer") and user_query:
    answer = retrieve_and_answer(user_query)
    st.write("**Answer:**")
    st.write(answer)

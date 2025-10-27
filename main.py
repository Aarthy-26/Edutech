
import streamlit as st
import json
from langchain_helper import create_vector_db, answer_with_groq

st.set_page_config(page_title="Ed-Tech Q&A", layout="centered")

st.title("Ed-Tech Q&A")

st.markdown(
    "Ask questions about the bootcamp or courses."
)

if st.button("Create Knowledgebase"):
    with st.spinner("Creating FAISS index from CSV (this may take a moment)..."):
        create_vector_db()
    st.success("Knowledge base created (faiss_index).")

question = st.text_input("Type your question here and press Enter")

if question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            answer, sources, raw = answer_with_groq(question, k=1)
        except Exception as e:
            st.error("Error while generating answer. See details below.")
            st.exception(e)
        else:
            if sources:
                for d in sources:
                    content = d.page_content.strip()
                    if "response:" in content and "prompt:" in content:
                        try:
                            # Extract only the response
                            response_part = content.split("response:", 1)[1].strip()
                            st.write(response_part)  # Just print the response text
                        except Exception:
                            st.write(content)
                    else:
                        st.write(content)


import streamlit as st
import pandas as pd
from backend import analyze

st.title("AI DATA ANALYST")

uploaded_files = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

tables = {}
for file in uploaded_files:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    tables[file.name] = df
    st.write("Table:", file.name)
    st.dataframe(df)

question = st.text_input("Ask your question")

if st.button("ANALYZE"):
    if not uploaded_files:
        st.error("Please upload a CSV or Excel file.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        result = analyze(tables, question)

        if result["can_determine"]:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Generated Python Code")
            st.code(result["code"], language="python")

            st.subheader("Evidence / Proof")
            st.write(result["evidence"])
        else:
            st.error("Cannot determine: " + result["reason"])
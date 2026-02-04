import streamlit as st
from document_processor import VectorStore

st.set_page_config(
    page_title="SmartDocQA",
    page_icon="📄",
    layout="centered"
)

def main():
    st.title("SmartDocQA")
    st.caption("Ask questions directly from your PDF documents using AI")


    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

    st.subheader("Ask a Question")
    user_query = st.text_input(
        "Type your question here",
        placeholder="e.g. What is this document about?"
    )

    if st.button("Get Answer"):
        if not uploaded_file or not user_query:
            st.warning("Please upload a PDF and enter a question.")
            return

        pdf_path = "uploaded_document.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        if "vectorstore" not in st.session_state:
            with st.spinner("Processing document (one-time)..."):
                st.session_state.vectorstore = VectorStore(pdf_path)

        with st.spinner("Generating answer..."):
            results = st.session_state.vectorstore.retrieve(user_query)

        st.subheader("Answer")

        for i, text in enumerate(results, start=1):
            st.markdown(
                f"""
                <div style="
                    background-color:#f6f8fa;
                    padding:15px;
                    border-radius:8px;
                    margin-bottom:10px;
                ">
                <b>Result {i}</b><br>
                {text}
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

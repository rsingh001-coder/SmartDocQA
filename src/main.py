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

    # ---------- Upload ----------
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # ---------- Track file change ----------
    if "last_file_name" not in st.session_state:
        st.session_state.last_file_name = None

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

    # ---------- Question ----------
    st.subheader("Ask a Question")
    user_query = st.text_input(
        "Type your question here",
        placeholder="e.g. What is this document about?"
    )

    # ---------- Button ----------
    if st.button("Get Answer"):
        if not uploaded_file or not user_query:
            st.warning("Please upload a PDF and enter a question.")
            return

        pdf_path = "uploaded_document.pdf"

        # 🔥 Rebuild VectorStore ONLY if new PDF uploaded
        if uploaded_file.name != st.session_state.last_file_name:
            st.session_state.last_file_name = uploaded_file.name

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Processing new document..."):
                st.session_state.vectorstore = VectorStore(pdf_path)

        # ---------- Retrieve ----------
        with st.spinner("Generating answer..."):
            results = st.session_state.vectorstore.retrieve(user_query)

        # ---------- UI ----------
        st.subheader("🤖 Answer")

        for i, text in enumerate(results, start=1):
            st.markdown(
                f"""
                <div style="
                    background-color:#f6f8fa;
                    color:#111827;
                    padding:16px;
                    border-radius:10px;
                    margin-bottom:12px;
                    font-size:15px;
                    line-height:1.6;
                ">
                <b>Result {i}</b><br><br>
                {text}
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()

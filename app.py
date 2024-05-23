from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()


#user interface with streamlit framework
#contents on sidebar

# with st.sidebar:
#     st.title('💬PDF Summarizer and Q/A App')
#     st.markdown('''
#                 ## About this application
#     You can built your own customized LLM-powered chatbot using:
#     - [Streamlit](https://streamlit.io/)
#     - [LangChain](https://python.langchain.com/)
#     - [OpenAI](https://platform.openai.com/docs/models) LLM model
#                 ''')       
#     add_vertical_space(2)
#     st.write('Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading.')
st.header(" PDF Summarizer QnA Bot 📑 🤖")
def main():
    os.environ["OPENAI_API_KEY"]= os.getenv("OPENAI_API_KEY")

    #upload a pdf file

    doc = st.file_uploader("Upload your pdf file here", type = "pdf")
    
    # extract the text
    if doc is not None:
        reader = PdfReader(doc)
        text = ""
        for page_text in reader.pages:
            if page_text:
                text += page_text.extract_text()

#split text into chunks

        text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
        data = text_splitter.split_text(text)


#create Embeddings 

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        database = FAISS.from_texts(data, embeddings)
# show user input
        
        user_question = st.text_input("Please ask a question about your PDF here:")
        if user_question:
            search_results = database.similarity_search(user_question)

#connect llm OpenAI


            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type = "stuff", verbose = True)
            with get_openai_callback() as cb:
                response = chain.run(input_documents=search_results, question = user_question)
                print(cb)
            st.write(response)

    
        
        
if __name__ == '__main__':
    main()
     

      
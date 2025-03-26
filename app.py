#This is the basic code for the streamlit app. This code will be used to create a web app that will take the user input and return the similar text from the dataset.
#The dataset is stored in the embeddings_data.csv file. The dataset contains the text data that will be used to find the similar text.
#from the dataset you can replace it with your own dataset.You can customize for some book search or any other search.
#The code will use the langchain library to find the similar text from the dataset.
#The code will use the HuggingFaceEmbeddings to find the embeddings of the text data and then use the FAISS to find the similar text from the dataset.


#importing libraries
import streamlit as st
import langchain_community
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings()


#setting the page configuration of the web app
st.set_page_config(page_title="Educate KIds", page_icon=":books:")
st.header("Hey ask me anything")

import pandas as pd
df = pd.read_csv("embeddings_data.csv")

df = df.drop(columns=['ID'])

#creating the document from the dataset

from langchain.docstore.document import Document
documents = []

for index, row in df.iterrows():
   
    documents.append(Document(page_content=row['Text'], metadata=row.to_dict()))

db = FAISS.from_documents(documents, embeddings)

def input_text():
  input_text = st.text_input("Ask me anything", key=input)
  return input_text
user_input = input_text()
submitted = st.button("Find Similar things")

if submitted:
  similar_docs = db.similarity_search(user_input)
  for i in similar_docs:
    st.write(i.page_content)

#after all this you will have to run the streamlit app using the command streamlit run App.py in the terminal.
#This will run the streamlit app and you can access the web app on the localhost.
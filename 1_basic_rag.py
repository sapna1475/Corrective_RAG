from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documnets import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import  List , TypedDict
from langgraph.graph  import StateGraph , START , END
from dotenv import load_dotenv

load_dotenv()

#Step 1: Fetch the documents
docs = (
    PyPDFLoader("./documents/book.pdf").load()
)

#step 2: Split the documents into chunks
chunks = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200).split_documents(docs)

#clean text to avoid unicode encoding issues
for d in chunks: 
    d.page_content = d.page_content.encode("utf-8", "ignore").decode()

#step 3: Embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-large')
vector_store = FAISS.from_documents(chunks, embeddings)

#Step 4 retriver-> top k documents
retriever = vector_store.as_retriever(search_type = 'similarity' , search_kwargs = {'k' : 4} )

#Step 5: llm
llm = ChatOpenAI(model = "gpt-4o-mini" , temperature = 0.1)

#state
class State(TypedDict):
    question: str
    docs : List[Document]
    answer: str

#retrive teh relevant docs/context based on the q
def retrieve(state):
    q = state['question']
    return {"docs" : retriever.invoke(q)}

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answer questions onlyfrom the context. If not in context, say you dont know."),
    ("human" "Question: {question}\n\nContext : \n{context}")
    
])
def generate(state):
    context = "\n\n".join(d.page_content for d in state['docs'])
    out = (prompt | llm.invoke({"question": state['question'] , "context" : context}))
    return {"answer" : out.content}

g = StateGraph(State)
g.add_node("retrieve" , retrieve)
g.add_node("generate" , generate)

g.add_edge(START , "retrieve")
g.add_edge("retrieve" , "generate")
g.add_edge("generate" , END)
app = g.compile()


#Run
res = app.invoke({"question" : "What is transformer architecture?" , "docs" : [] , "answer" : ""})
print(res['answer'])


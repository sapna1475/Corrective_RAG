from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documnets import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import  List , TypedDict
from langgraph.graph  import StateGraph , START , END
from dotenv import load_dotenv
from pydantic import BaseModel

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

    strips : List[str]    #output of decomposition(sentence strips)
    kept_strips : List[str]    #after filtering - strips that are relevant
    refined_context : str   # recompose internal knowledge(join kept_strips)


#retrive teh relevant docs/context based on the q
def retrieve(state):
    q = state['question']
    return {"docs" : retriever.invoke(q)}


# -------------------------------------------------------------------
# Sentence-level DECOMPOSER
# -------------------------------------------------------------------
def decompose_to_sentences(text: str) -> List[str]:
    """
    Decomposes the input text into a list of sentences.
    """
    text = re.sub(r"\s+" , " " , text).strip()   #remove extra whitespace
    sentences = re.split(r'(?<=[.!?])\s+' , text)  #split by sentence-ending punctuation
    return [s.stip() for s in sentences if len(s.stip()) > 20 ]    

# ----------------------------------------------------------------------------
# FILTER (LLM judge)
# --------------------------------------------------------------------------
#pydantic schemafor the output for filter chain to pass with the llm 
class keepOrDrop(BaseModel):
    keep : bool

#prompt for filtering sentences
filter_prompt=  ChatPromptTemplate.from_messages([

    ("system" , "You are a strict relevance filter.\n"
    "Return keep = true only if the sentence directly helps answer the question. \n"
    "Use ONLY the sentence. Output JSON only."
    ), 
    ("human" , "Question: {question}\n\nSentence: {sentence}")
])

filter_chain = filter_prompt | llm.with_structured_output(keepOrDrop)

# ------------------------------------------------------------------------
# REFINING (Decompose -> Filter -> Recompose)
# -------------------------------------------------------------------------
def refine(state : State) -> State:
    q = state["question"]

    #combine retrieved docs into a single context
    context = "\n\n".join(d.page_content for d in state['docs']).strip()

    #1. Decompose context into sentences 
    strips = decompose_to_sentences(context)

    #2. Filter sentences using LLM
    kept : List[str] = []
    for s in strips:
        if filter_chain.invoke({"question" : q , "sentence" : s}).keep:
            kept.append(s)

    #3. Recompose the kept strips together for the refined context
    refined_context = "\n".join(kept).strip()

    return {
        "strips" : strips,
        "kept_strips" : kept,
        "refined_context" : refined_context,
    }

#Final LLM prompt to generate the response
answer_prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a helpful tutor. Anwer only using the provided bullets. \n"
    "If the bullets are empty or insufficient, say I dont know. "
    ),
    ("huma", "Question: {question}\n\n Refined context : \n{refined_context}")

])

def generate(state: State) -> State : 
    out = (answer_prompt | llm.invoke({"question" : state['question'] , "refined_context" : state['refined_context']}))
    return {"answer" : out.content}



g = StateGraph(State)
g.add_node("retrieve" , retrieve)
g.add_node("refine" , refine)
g.add_node("generate" , generate)

g.add_edge(START , "retrieve")
g.add_edge("retrieve" , "refine")
g.add_edge("refine" , "generate")
g.add_edge("generate" , END)
app = g.compile()


#invoke the model
res = app.invoke({
    "question": "Explain the bias–variance tradeoff",
    "docs": [],
    "strips": [],
    "kept_strips": [],
    "refined_context": "",
    "answer": ""
})
print(res["answer"])
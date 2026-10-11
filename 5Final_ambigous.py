from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documnets import Document
from langchain_core.prompts import ChatPromptTemplate
from typing import  List , TypedDict , Literal
from langgraph.graph  import StateGraph , START , END
from dotenv import load_dotenv
from pydantic import BaseModel

#Tavily
from langchain_community.tools.tavily_search import TavilySearchResults

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

UPPER_TH = 0.7
LOWER_TH = 0.3

#state
class State(TypedDict):
    question: str
    docs : List[Document]

    #after retrieval eval
    good_docs : List[Document]
    verdict : str
    reason : str

    strips : List[str]    #output of decomposition(sentence strips)
    kept_strips : List[str]    #after filtering - strips that are relevant
    refined_context : str   # recompose internal knowledge(join kept_strips)

    #web search documents 
    web_docs : List[Document]

    answer : str

#retrive the relevant docs/context based on the q
def retrieve(state):
    q = state['question']
    return {"docs" : retriever.invoke(q)}



# --------------------------------------------------------------------------
# Score-based retrieved doc evaluator
# --------------------------------------------------------------------------

#pydantic schema for the mdoel to stricly follow the output format
class DocEvalScore(BaseModel):
    score: float
    reason : str

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a strict retrieval evaluator for RAG\n"
    "You will be given ONE retrieved chunk and a question\n"
    "Return a relevance score in [0.0 1.0] . \n"
    "1.0: chunk alone is sufficient to answer fully.\n"
    "0.0: chunk is irrelevant.\n"
    "Be conservative with high scores. \n"
    "Also return a short reason for the score. \n"
    "Output JSON only."
    )
    ("human" , "Question : {question}\n\nChunk: {chunk}")
])

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)

def eval_each_doc_node(state: State) -> State: 
    q = state["question"]
    scores: List[float] = []
    reasons : List[str] = []
    good : List[Document] = []

    for d in state["docs"]: 
        out = doc_eval_chain.invoke({"question" : q, "chunk" : d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)

        #1Correct case we will refine docs
        if out.score > LOWER_TH: 
            good.append(d)

    #2 correct if at leasts one doc > Upper_th
    if any(s > UPPER_TH for s in scores): 
        return {
            "good_docs" : good, 
            "verdict" : "CORRECT", 
            "reason" : "At least one doc is highly relevant"
        }
    #3 INCORRECT if all docs < LOWER_TH
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        return {
            "good_docs" : [],
            "verdict" : "INCORRECT",
            "reason" : "All docs are irrelevant"
        }
    #4 else all are ambigous if none are > upper and lower than < lower_th
    why = "All docs are ambigous. "
    return {
        "good_docs" : good, 
        "verdict" : "AMBIGUOUS",
        "reason": why + " ".join(reasons)
    }


#Knowledge Refinement 
# -------------------------------------------------------------------
# Sentence-level DECOMPOSER
# -------------------------------------------------------------------
def decompose_to_sentences(text: str) -> List[str]:
    """
    Decomposes the input text into a list of sentences.
    """
    text = re.sub(r"\s+" , " " , text).strip()   #remove extra whitespace
    sentences = re.split(r'(?<=[.!?])\s+' , text)  #split by sentence-ending punctuation
    return [s.strip() for s in sentences if len(s.strip()) > 20 ]    

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

    #combine retrieved docs into a single context from the good docs only
    if state.get("verdict") == "CORRECT":
        #build context from good docs only
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state['web_docs']
    else:
        #Ambigous
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n" .join(d.page_content for d in docs_to_use).strip()
   

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


# -------------------------------------------------------------------------
# Web search (Iteration 4)
# Assumption: web search does not fail (no fail node in this branch)
# ---------------------------------------------------------------------------
tavily = TavilySearchResults(max_results = 5)

def web_search_node(state : State) -> State:
    q = state["question"] 
    results = tavily.invoke({"query" : q})

    web_docs = []
    for r in results or []:
        title = r.get("title" , "")
        url = r.get("url" , "")
        content = r.get("content", "") or r.get("snippet", "")
        
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}




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



#Routing
#CORRECT - refine
#INCORRECT / AMBIGOUS - web search - refine - generate

def route_after_eval(state : State)-> State: 
    if state["verdict"] == "CORRECT": 
        return "refine"
    
    else:
        return "web_search"



g = StateGraph(State)
g.add_node("retrieve" , retrieve)
g.add_node("eval_each_doc" ,eval_each_doc_node )
g.add_node("web_search" , web_search_node)
g.add_node("refine" , refine)
g.add_node("generate" , generate)



g.add_edge(START , "retrieve")
g.add_edge("retrieve" , "eval_each_doc")
#conditional edge
g.add_conditional_edges("eval_each_doc" ,
     route_after_eval, 
     {"refine" : "refine ", "web_search" : "web_search"} )


g.add_edge("web_search" , "refine")
g.add_edge("refine" , "generate")


g.add_edge("generate" , END)
app = g.compile()


#invoke the model
# -----------------------------
# Run example
# -----------------------------
res = app.invoke(
    {
        "question": "Batch normalization vs layer normalization",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }
)

print("VERDICT:", res["verdict"])
print("REASON:", res["reason"])
print("WEB_QUERY:", res["web_query"])
print("\nOUTPUT:\n", res["answer"])
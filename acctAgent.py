from typing import Tuple, Dict, Any
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import BaseTool, FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_parse import LlamaParse
from llama_index.core.agent import ReActAgent
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.agent import FnAgentWorker
from constants import *
from prompt import *


parser = LlamaParse(api_key=parse_key, result_type="markdown")

extractor = {".csv": parser}
# set the default tim as gemini
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=API_KEY)
gemini_embedding = GeminiEmbedding(model="models/gemini-1.5-flash", api_key=API_KEY)
Settings.embed_model = gemini_embedding


docs = SimpleDirectoryReader("price_list", file_extractor=extractor).load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("categorise product by category")
print(response)

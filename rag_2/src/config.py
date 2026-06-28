from pathlib import Path


# Define hyperparameters
CONFIG = {
	"experiment_name": "rag_2 + cohere_reranking + hybrid + metadata",
	"pdf_parser": "pymypdf4llm",
	"splitter": "MarkdownHeaderTextSplitter",
	"headers_to_split_on": [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")],
	"chunk_size": None,
   "chunk_overlap": None,
	"embedding_model": "text-embedding-3-large",
	"embedding_size": 3072,
	"vdb_top_k": 10,
	"bm25_k": 10,
	"cohere_k": 3
}


# Define paths
BASE_DIR = Path(__file__).resolve().parent

APPLE_DOC_PATH = list(Path.cwd().rglob("Apple_form_10k.pdf"))
FINANCIAL_DATA_PATH = list(Path.cwd().rglob("Sample_Financial_Data.xlsx"))

VDB_PATH = BASE_DIR / 'data' / 'vectorstore'
BM25S_PATH = str(BASE_DIR / 'data' / 'bm25s_index')

GD_PATH = str(BASE_DIR / 'eval' / "golden_dataset.jsonl")
RETRIEVAL_TEST_RESULTS_PATH = BASE_DIR /  'eval' / 'retrieval_test_results.jsonl'
RETRIEVAL_LOG_PATH = BASE_DIR / 'eval' / 'retrieval_logs.csv'
CLASSIC_RETRIEVAL_MATRICES_PATH = BASE_DIR / 'eval' / 'classic_retrieval_logs.csv'

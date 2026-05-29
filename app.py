from fileinput import filename

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Security, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
# Suppress Hugging Face warning and info logs before any ML imports occur
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from dotenv import load_dotenv

load_dotenv()

from pipeline.rag import rag_model
import logging
import shutil
import pynvml 
import time
import tiktoken
from pathlib import Path

# --- NEW targeted LOGGER SETUP ---
# 1. Create a isolated named logger instead of utilizing basicConfig
logger = logging.getLogger("RAG_METRICS")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevents logs from bubbling up to root system logs

# 2. Define the exact text format layout
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# 3. Create file handler targeting only your metrics file
file_handler = logging.FileHandler("rag_system.log", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 4. Create console handler to mirror to terminal screen
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 5. SILENCE THE THIRD-PARTY CHATTER (Uvicorn, Watchfiles, and Transformers)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles").setLevel(logging.WARNING)     # Silences "1 change detected"
logging.getLogger("transformers").setLevel(logging.ERROR)   # Silences HuggingFace chatter

env_keys = os.getenv("VALID_API_KEYS", "")
VALID_API_KEYS = set(env_keys.split(",")) if env_keys else set()
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
tokenizer_proxy = tiktoken.get_encoding("cl100k_base")    

app = FastAPI(title="RAG Model API", description="Production API for PDF Question Answering")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://burning-river.github.io"
    ],
    allow_origin_regex="file://.*", 
    allow_credentials=False,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],  # Allows all headers (Content-Type, etc.)
)
# Initialize NVML for exact GPU tracking
try:
    pynvml.nvmlInit()
    gpu_available = True
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Tracks GPU 0
except Exception:
    gpu_available = False
    logger.warning("NVIDIA GPU Driver/NVML not found. Falling back to CPU logging status.")

def get_gpu_memory_used():
    """Returns currently allocated GPU memory in Megabytes (MB)."""
    if gpu_available:
        info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        return round(info.used / (1024 ** 2), 2)  # Convert bytes to MB
    return 0.0

def cleanup_temp_file(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"--> Successfully cleaned up temp file: {file_path}")

print("Loading RAG models and embeddings into memory...")

# @app.post("/api/verify")
# async def validate_api_key(api_key: str = Security(api_key_header)):
#     """Validates the presence and authenticity of the X-API-Key header."""
#     if not api_key:
#         raise HTTPException(
#             status_code=401, 
#             detail="Authentication credentials missing. Please provide an X-API-Key header."
#         )
#     if api_key not in VALID_API_KEYS:
#         raise HTTPException(
#             status_code=403, 
#             detail="Access Denied: Invalid API Key provided."
#         )
#     return api_key

@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, # Inject the FastAPI background task tool
                      file: UploadFile = File(...),
                    #   authenticated_key: str = Depends(validate_api_key)
                      ):

    # storage_dir = Path("./data/rag_files") 
    # storage_dir.mkdir(parents=True, exist_ok=True)
    # file_path = storage_dir / file.filename
    os.makedirs("/tmp/rag_files", exist_ok=True)
    file_path = f"/tmp/rag_files/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print('file uploaded and saved to disk at:', file_path)
    
    return {"message": "File uploaded successfully", "filename": file.filename}

@app.post("/api/query")
async def query_pdf(
    filename: str = Form(..., description="The PDF document to analyze"),
    query: str = Form(..., description="The question you want to ask the RAG model"),
):
    
    # storage_dir = Path("./data/rag_files")
    # temp_file_path = storage_dir / filename
    file_path_str = f"/tmp/rag_files/{filename}"

    # file_path_str  = str(temp_file_path.resolve())

    # 1. Validate file type
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    start_time = time.time()
    # logger.info(f"Authenticated Request (Key: Token...{authenticated_key[-4:]}) -> File: {filename}")
    gpu_start_mem = get_gpu_memory_used()
    print('--- Starting RAG processing pipeline ---')

    async def stream_generator():
        full_response_text = []
        # Start tracking time immediately when streaming starts
        api_start_time = time.time()
        
        try:
            async for token in rag_model(file_path_str, query):
                full_response_text.append(token)
                yield token
        finally:
            # 1. Compute End-to-End API Latency
            latency = round(time.time() - api_start_time, 2)
            final_answer = "".join(full_response_text)
            
            # 2. Compute Token Metrics via Proxy Tokenizer (No Local GPU Model Needed)
            input_tokens = len(tokenizer_proxy.encode(query))
            output_tokens = len(tokenizer_proxy.encode(final_answer))
            total_tokens = input_tokens + output_tokens
            
            # 3. Compute Network Generation Performance Metrics
            tokens_per_second = round(output_tokens / latency, 2) if latency > 0 else 0.0

            # 4. LOG COMPREHENSIVE PRODUCTION METRICS (GPU Removed)
            logger.info(
                f"\n=== TRANSACTION PERFORMANCE REPORT ===\n"
                # f"Tx Success | Key Match: ...{authenticated_key[-4:]}\n"
                f"File Processed     : {filename}\n"
                f"API Round Trip     : {latency} seconds\n"
                f"Network Throughput : {tokens_per_second} tokens/sec\n"
                f"----------------------------------------\n"
                f"Estimated Prompt   : {input_tokens} tokens\n"
                f"Estimated Generated: {output_tokens} tokens\n"
                f"Estimated Total    : {total_tokens} tokens\n"
                f"========================================"
            )

    # Return the clean streaming response
    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    # Force the app to load as a strict import string string
    port = int(os.getenv("PORT", 8000)) 

    uvicorn.run("app:app", host="0.0.0.0", port=port)
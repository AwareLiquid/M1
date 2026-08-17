# MT-LNN native inference server — CPU image.
#
# Build:
#   docker build -t mt-lnn-serve .
#
# Run (fresh untrained model, byte tokenizer — smoke):
#   docker run --rm -p 8000:8000 -e SMALL=1 mt-lnn-serve
#
# Run (trained checkpoint + GPT-2 tokenizer):
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -e CKPT_PATH=/app/checkpoints/final.pt -e TOKENIZER=gpt2 \
#     mt-lnn-serve
#
# For CUDA, swap the base image for an nvidia/cuda runtime and install the
# matching torch wheel, then `docker run --gpus all ... -e DEVICE=cuda`.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# CPU-only torch first (smaller, no CUDA libs), then the rest.
# NB: quote every "pkg>=ver" — an unquoted '>' is a shell redirection.
COPY serve/requirements-serve.txt /app/serve/requirements-serve.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
 && pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.27.0" "pydantic>=2.0.0" \
        "transformers>=4.40.0" "tokenizers>=0.15.0" "peft>=0.10.0"

# App code (mt_lnn package + serve entrypoint).
COPY mt_lnn /app/mt_lnn
COPY serve /app/serve

EXPOSE 8000

# Container healthcheck hits the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "serve.server:app", "--host", "0.0.0.0", "--port", "8000"]

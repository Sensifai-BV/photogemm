# Use the official vLLM image, which includes CUDA, PyTorch, and vLLM
FROM vllm/vllm-openai:latest
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your models and templates
COPY ./lora_models /app/lora_models
COPY ./tool_chat_template_gemma3_pythonic.jinja .

EXPOSE 8000

# Default fallback (overridden by docker-compose)
CMD ["vllm", "serve", "google/gemma-3-4b-it"]
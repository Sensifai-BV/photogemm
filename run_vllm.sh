vllm serve /home/ubuntu/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767 \
    --chat-template tool_chat_template_gemma3_pythonic.jinja \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules my_gemma_lora=./lora_models/train_2025-10-04-12-58-31 \
                   my_lora_boolean=./lora_models/train_2025-10-12-11-16-24-bool2 \
                   my_scoring_lora=./lora_models/train_2025-10-15-06-00-49-finale1 \
                   

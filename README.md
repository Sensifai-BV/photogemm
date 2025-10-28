# deploy_gemma_vllm



## Getting started

clone the repository using whether ssh or https.

## Install requirements

```
conda create -n vllm_env python=3.10
conda activate vllm_env
pip install -r requirements.txt
```

## Copy Lora models
Copy all lora models in lora_models directory

## Run Vllm engine
This repo includes both vllm engine and main app. the vllm engine which will run your models on the port 8000 became loaded by:

```
chmod 700 run_vllm.sh 
./run_vllm.sh
```
** Note that is it better to run your runner in tmux!

## Run main app
After running your engine, you can run your main app

```
python app.py
```

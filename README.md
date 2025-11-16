# PHOTOGEMM


## Getting started

clone the repository using whether ssh or https.

### Copy Lora models
Copy all lora models in lora_models directory

## Manually Run Photogemm (Not Recommended)
#### Install requirements

```
conda create -n photogemm python=3.10
conda activate photogemm
pip install -r requirements.txt
```


#### Run Vllm engine bash file
This repo includes both vllm engine and main app. the vllm engine which will run your models on the port 8000 became loaded by:

```
chmod 700 run_vllm.sh 
./run_vllm.sh
```
** Note that is it better to run your runner in tmux!

## Build Photogemm Docker (Recommended)
There is a dockerFile exist in the project which relates to 
PhotoGemm image. This image is to handlig the vllm dependecies and run the code in its container. All you need is to run photogemm docker compose file:

```
docker compose -f docker-compose-photogemm.yml up -d
```

This will build the image and run the vllm engine on the 8000 port of your LM.

## Run PhotogemmQ

To run the quantized version of Photogemm which is named PhotogemmQ, you should build and run it using llamacpp. Since llamacpp does not compativle with the all versions of cuda and you should face cuda mismatch, there is a specific image to run llamacpp. In that image llamacpp is built and is ready to be used.
There are some steps you should take before running photogemmq's docker compose file. 

First, go to 'docker-compose-photogemmq.yml'. In the line 12, replace 'MODEL_PATH' with the path where your quantized model is exist.

Then, replace the model and mmproj names in the lines 19 and 21 with your model's name. 

Now, you are free to run the photogemmq:

```
docker compose -f docker-compose-photo_gemmq.yml up -d
```


## Run main apps for the endpoints

There are 2 fastapi python apps to run photogemm and photogemmQ. Both files will be run on the  8005 endpoint. For photogemmQ there is only a single call in this version. But, by running 'photo_gemm_app.py' there will be endpoints for both multiple call and single call. You can checkout the endpoints there. 

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Form
import uvicorn
import base64
import io
import time
import os
import httpx
import pickle
import asyncio
from PIL import Image
from openai import OpenAI
import json
import logging
import json
from enum import Enum, auto
from handle_non_challenging_request import NonChallegingHandler
from parameters_naming import ModelConfig
from preprcessors import GoldenRectangle, RuleOfThird

# SAVE_DIR = './uploaded _photos'

lora_params = [
    'soft_focus',
    'hue',
    'dust_visibility',
    'point_of_light',
]

boolean_lora_params = [
    'leading_space',
    'leading_lines',
    'pattern_recognition',
    'retouching',
    'symmetrical_balance',
    'frame_in_frame',
    'haze_presence',
    'diagonal_leading_lines',
    'colormode',
    'double_exposure',
    'hdr',
    'lens_flare'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Analysis API", version="1.0.0")

# OpenAI client configuration for your vLLM server
openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

non_challenging_handler = NonChallegingHandler()
my_configs = ModelConfig()

rule_of_thirds_preprocessor = RuleOfThird()
golden_rectangle_preprocessor = GoldenRectangle()



def image_to_base64_data_url(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Image Analysis API is running"}




@app.post("/analyze-image-single")
async def analyze_image_single(
    parameter_choice: int,
    file: UploadFile = File(...),
):
    """
    Analyze parameters in an uploaded image
    
    Args:
        file: Uploaded image file
        instruction: Custom instruction text for the analysis
        model_choice: Model to use (base, lora, or merged)
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Minimizing Distractions
    2. Subtle Complexity
    3. Composition Contrast
    4. Depth of Field
    5. Negative Space
    6. Texture
    7. Aspect Ratio
    8. Implied Lines
    9. Scale and Proportion
    10. Color Harmony
    11. Location Setting
    12. Rule of Thirds
    13. Haze Impact
    14. Dust Existence
    15. Fringe
    16. Starburst
    17. Moiré
    18. Luminosity
    19. Saturation
    20. Contrast
    21. Sharpness
    22. Light Source Viewer
    23. Color Temperature
    24. Visual Harmony
    25. Overall Expert Assessment
    26. Overall Technical Assessment
    27. Development
    28. Image Realism
    29. AI Creation Probability
    30. Photographic Intent
    31. Emotional Impact
    32. Intended Purpose
    33. Target Audience
    34. Detailed Emotional Impact
    35. Color Mode
    36. Leading Space
    37. Leading Lines
    38. Pattern Recognition
    39. Retouching
    40. Symmetrical Balance
    41. Frame in Frame
    42. Haze Presence
    43. Diagonal Leading Lines
    44. Double Exposure
    45. HDR
    46. Lens Flare
    47. Exposure
    48. Perspective Shift
    49. Perspective Lines
    50. Light Softness Viewer
    51. Light Reflection
    52. Pattern Repetition
    53. Point of Light
    54. Sense of Motion
    55. Dust Visibility
    56. Soft Focus
    57. Digital Noise\n
    58. Hue
    59. Golden Rectangle
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        
        if parameter_choice == 12:
            processed_image = rule_of_thirds_preprocessor.process(data_input=Image.open(file.file))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG") 
            image_data = byte_buffer.getvalue()
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            choosen_param = "Rule of Thirds"
            instruction = f"Analyze the image for the 'tule of thirds' parameter and provide the result as a JSON object."
            model = my_configs.get_rot_model_path()
        
        elif parameter_choice == 59:
            processed_image = golden_rectangle_preprocessor.process(data_input=Image.open(file.file))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG")
            image_data = byte_buffer.getvalue()
            choosen_param = "Golden Rectangle"
            instruction = f"Analyze the image for the 'golden rectangle' parameter and provide the result as a JSON object."
            model = my_configs.get_gr_model_path()
            
        elif parameter_choice < 36 or parameter_choice == 46 or parameter_choice == 42:
            # if parameter_choice == 40: parameter_choice = 36
            if parameter_choice == 46: parameter_choice = 37
            if parameter_choice == 42: parameter_choice = 38
            
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            model = my_configs.get_base_model_path()
        
        elif parameter_choice > 35 and parameter_choice < 47:
            choosen_param = my_configs.get_normal_bool_names()[parameter_choice-36]
            instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
            model = my_configs.get_boolean_lora_path()
            
        else:
            if parameter_choice == 58:
                choosen_param = "Hue"
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                model = my_configs.get_lora_model_path()
            else:
                choosen_param = my_configs.get_normal_scoring_names()[parameter_choice-47]
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                model = my_configs.get_scoring_lora_path()
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        

        logger.info(f"Processing im`age: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {choosen_param}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": file.filename or "uploaded_image"
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = choosen_param
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/analyze-image-multiple")
async def analyze_image_multiple(
    parameter_choices: str,  # Comma-separated list like "1,5,16,19,27,32,39,40,55,57,58"
    file: UploadFile = File(...),
):
    """
    Analyze multiple parameters in an uploaded image asynchronously
    
    Args:
        parameter_choices: Comma-separated string of parameter indices (e.g., "1,5,16,19")
        file: Uploaded image file
        
    Returns:
        JSON response with all analyses from different models
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Parse parameter choices
        try:
            param_list = [int(p.strip()) for p in parameter_choices.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid parameter format. Use comma-separated numbers.")
        
        # Validate parameter ranges
        if any(p < 1 or p > 58 for p in param_list):
            raise HTTPException(status_code=400, detail="Parameters must be between 1 and 58")
        
        # Read and prepare image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        # Save image
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
        
        logger.info(f"Processing image: {file.filename} with parameters: {param_list}")
        
        # Group parameters by model
        base_params = []           # 1-35 - each needs individual call
        boolean_params = []        # 36-46 - batched
        scoring_params = []        # 47-57 - batched
        hue_params = []           # 58 - batched
        
        for param in param_list:
            if param < 36:
                base_params.append(param)
            elif 36 <= param < 47:
                boolean_params.append(param)
            elif param == 58:
                hue_params.append(param)
            else:  # 47-57
                scoring_params.append(param)
        
        
        async def call_base_model_batch(params_batch):
            """Call base model for a batch of k parameters"""
            model_path = my_configs.get_base_model_path()
            
            # Collect instructions and parameter names
            instructions = []
            param_names = []
            
            for param in params_batch:
                instruction, param_name = non_challenging_handler(param)
                instructions.append(instruction)
                param_names.append(param_name)
            
            # Merge instructions
            merged_instruction = "\n".join(instructions)
            merged_instruction += f"\n\nReturn the response for the {len(params_batch)} given parameters in a single dictionary containing name, result and explanation for each."
            
            logger.info(f"Calling base model batch for parameters: {param_names}")
            logger.info(f"Merged instruction: {merged_instruction[:200]}...")
            
            chat_response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": merged_instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            },
                            "uuid": file.filename or "uploaded_image"
                        },
                    ],
                }],
                temperature=0
            )
            
            response_content = chat_response.choices[0].message.content
            logger.info(f"Received response for batch: {param_names}")
            
            try:
                if "```json" in response_content:
                    json_str = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    json_str = response_content.split("```")[1].strip()
                else:
                    json_str = response_content
                
                parsed_response = json.loads(json_str)
                return {
                    "group": "base_model_batch",
                    "model": model_path,
                    "parameters": param_names,
                    "parameter_indices": params_batch,
                    "batch_size": len(params_batch),
                    "analysis": parsed_response,
                    "raw_response": response_content
                }
            except json.JSONDecodeError:
                return {
                    "group": "base_model_batch",
                    "model": model_path,
                    "parameters": param_names,
                    "parameter_indices": params_batch,
                    "batch_size": len(params_batch),
                    "raw_response": response_content,
                    "note": "Response could not be parsed as JSON"
                }
        
        # Helper function for batched model calls (boolean, scoring, hue)
        async def call_batched_model(params, model_path, group_name, name_getter):
            """Call model with multiple parameters in batch"""
            if not params:
                return None
            
            # Build parameter names
            param_names = [name_getter(param) for param in params]
            
            # Create instruction for multiple parameters
            param_list_str = ", ".join(param_names)
            instruction = f"Analyze the image for the following parameters: {param_list_str}. Provide the results as a JSON object with each parameter as a key."
            
            logger.info(f"Calling {group_name} with parameters: {param_names}")
            logger.info(f"Instruction: {instruction[:100]}...")
            
            chat_response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            },
                            "uuid": file.filename or "uploaded_image"
                        },
                    ],
                }],
                temperature=0
            )
            
            response_content = chat_response.choices[0].message.content
            logger.info(f"Received response from {group_name}")
            
            try:
                if "```json" in response_content:
                    json_str = response_content.split("```json")[1].split("```")[0].strip()
                elif "```" in response_content:
                    json_str = response_content.split("```")[1].strip()
                else:
                    json_str = response_content
                
                parsed_response = json.loads(json_str)
                return {
                    "group": group_name,
                    "model": model_path,
                    "parameters": param_names,
                    "parameter_indices": params,
                    "analysis": parsed_response,
                    "raw_response": response_content
                }
            except json.JSONDecodeError:
                return {
                    "group": group_name,
                    "model": model_path,
                    "parameters": param_names,
                    "parameter_indices": params,
                    "raw_response": response_content,
                    "note": "Response could not be parsed as JSON"
                }
        
        # Create async tasks
        tasks = []
        
        # Base parameters - each gets its own task
        # for param in base_params:
        #     tasks.append(call_base_model(param))
        base_batch_size = 18
        if base_params:
            # Split base_params into batches of size k
            for i in range(0, len(base_params), base_batch_size):
                batch = base_params[i:i + base_batch_size]
                tasks.append(call_base_model_batch(batch))
        
        # Boolean parameters - batched
        if boolean_params:
            tasks.append(call_batched_model(
                boolean_params,
                my_configs.get_boolean_lora_path(),
                "boolean_lora",
                lambda p: my_configs.get_normal_bool_names()[p-36]
            ))
        
        # Scoring parameters - batched
        if scoring_params:
            tasks.append(call_batched_model(
                scoring_params,
                my_configs.get_scoring_lora_path(),
                "scoring_lora",
                lambda p: my_configs.get_normal_scoring_names()[p-47]
            ))
        
        # Hue parameter - batched (though typically just one)
        if hue_params:
            tasks.append(call_batched_model(
                hue_params,
                my_configs.get_lora_model_path(),
                "hue_lora",
                lambda p: "Hue"
            ))
        
        # Execute all tasks concurrently
        start_time = time.perf_counter()  # Start timing

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()  # End timing
        elapsed_time = end_time - start_time

        # Filter out None results and handle exceptions
        successful_results = []
        errors = []

        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                logger.error(f"Error in model call: {result}")
            elif result is not None:
                successful_results.append(result)

        # Log latency and task count
        logger.info(
            f"Executed {len(tasks)} tasks in {elapsed_time:.2f} seconds "
            f"({len(successful_results)} successful, {len(errors)} failed)."
        )

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "requested_parameters": param_list,
            "results": successful_results,
            "errors": errors if errors else None,
            "total_tasks_executed": len(tasks),
            "successful_results": len(successful_results),
            "execution_time_sec": round(elapsed_time, 2)  # Optional: include in API response
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/analyze-image-url")
async def analyze_image_url(
    parameter_choice: int,
    image_url: str,
):
    """
    Analyze parameters in an image from a URL
    
    Args:
        image_url: URL of the image to analyze
        parameter_choice: Index of the parameter to analyze (1-58)
        
    Returns:
        JSON response with single analysis
        
    You should only enter the index of the parameter you are looking for:
    1. Minimizing Distractions
    2. Subtle Complexity
    3. Composition Contrast
    4. Depth of Field
    5. Negative Space
    6. Texture
    7. Aspect Ratio
    8. Implied Lines
    9. Scale and Proportion
    10. Color Harmony
    11. Location Setting
    12. Rule of Thirds
    13. Haze Impact
    14. Dust Existence
    15. Fringe
    16. Starburst
    17. Moiré
    18. Luminosity
    19. Saturation
    20. Contrast
    21. Sharpness
    22. Light Source Viewer
    23. Color Temperature
    24. Visual Harmony
    25. Overall Expert Assessment
    26. Overall Technical Assessment
    27. Development
    28. Image Realism
    29. AI Creation Probability
    30. Photographic Intent
    31. Emotional Impact
    32. Intended Purpose
    33. Target Audience
    34. Detailed Emotional Impact
    35. Color Mode
    36. Leading Space
    37. Leading Lines
    38. Pattern Recognition
    39. Retouching
    40. Symmetrical Balance
    41. Frame in Frame
    42. Haze Presence
    43. Diagonal Leading Lines
    44. Double Exposure
    45. HDR
    46. Lens Flare
    47. Exposure
    48. Perspective Shift
    49. Perspective Lines
    50. Light Softness Viewer
    51. Light Reflection
    52. Pattern Repetition
    53. Point of Light
    54. Sense of Motion
    55. Dust Visibility
    56. Soft Focus
    57. Digital Noise
    58. Hue
    """
    try:
        # Download image from URL
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
        # Verify it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL must point to an image")
        
        # Determine parameter and model based on parameter_choice
        if parameter_choice < 36:
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            model = my_configs.get_base_model_path()
        
        elif parameter_choice > 35 and parameter_choice < 47:
            choosen_param = my_configs.get_normal_bool_names()[parameter_choice-36]
            instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
            model = my_configs.get_boolean_lora_path()
            
        else:
            if parameter_choice == 58:
                choosen_param = "Hue"
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                model = my_configs.get_lora_model_path()
            else:
                choosen_param = my_configs.get_normal_scoring_names()[parameter_choice-47]
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                model = my_configs.get_scoring_lora_path()
        
        # Process image
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        # Extract filename from URL for logging and saving
        filename = image_url.split('/')[-1].split('?')[0] or "url_image.jpg"
        # Ensure filename has an extension
        if '.' not in filename:
            filename += '.jpg'
        
        logger.info(f"Processing image from URL: {image_url} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {choosen_param}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")
        
        # Save image to SAVE_DIR
        # save_path = os.path.join(SAVE_DIR, filename)
        # with open(save_path, "wb") as f:
        #     f.write(image_data)
        # logger.info(f"Image saved to: {save_path}")
        
        print(instruction)

        # Call vLLM API
        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": filename
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        # Parse JSON response
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            
            parsed_response = json.loads(json_str)
            parameter_value = choosen_param
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except httpx.HTTPError as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/analyze-image-mix-ft" , deprecated=True)
async def analyze_image_mix_ft(
    parameter_choice: str,
    file: UploadFile = File(...),
):
    """
    **DEPRECATED:** This endpoint is no longer maintained. 
    Please use the new `/analyze-image_mix-scoring` endpoint instead.
    Analyze parameters in an uploaded image
    
    Args:
        file: Uploaded image file
        instruction: Custom instruction text for the analysis
        model_choice: Model to use (base, lora, or merged)
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Soft Focus
    2. Hue
    3. Dust Visibility
    4. Point of Light

    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        input_values = parameter_choice.split(',') 
        parameters_list = [int(s.strip()) for s in input_values]
        

        chosen_params = ''
        for param in parameters_list:
            if chosen_params == "":
                chosen_params = my_configs.get_lora_params()[param-1]
            else:
                chosen_params = chosen_params + f', {my_configs.get_lora_params()[param-1]}'
        instruction = f"Analyze the image for {chosen_params}."
        model = my_configs.get_lora_model_path()
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        

        logger.info(f"Processing im`age: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {chosen_params}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": file.filename or "uploaded_image"
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = chosen_params
            full_response = {'Parameter': parameter_value, **parsed_response}            
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")




@app.post("/analyze-image-mix-boolean")
async def analyze_image_mix_boolean(
    parameter_choice: str,
    file: UploadFile = File(...),
):
    """
    Analyze parameters in an uploaded image
    
    Args:
        file: Uploaded image file
        instruction: Custom instruction text for the analysis
        model_choice: Model to use (base, lora, or merged)
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Leading Space
    2. Leading Lines
    3. Pattern Recognition
    4. Retouching
    5. Symmetrical Balance
    6. Frame in a frame
    7. Haze Presence
    8. Diagonal Leading Lines
    9. Double Exposure
    10. HDR\n
    11. Lens Flare

    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        input_values = parameter_choice.split(',') 
        parameters_list = [int(s.strip()) for s in input_values]
        

        chosen_params = ''
        for param in parameters_list:
            if chosen_params == "":
                chosen_params = my_configs.get_boolean_lora_params()[param-1]
            else:
                chosen_params = chosen_params + f', {my_configs.get_boolean_lora_params()[param-1]}'
        instruction = f"Analyze the image for {chosen_params}."
        model = my_configs.get_boolean_lora_path()
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        

        logger.info(f"Processing im`age: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {chosen_params}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": file.filename or "uploaded_image"
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = chosen_params
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/analyze-image-mix-boolean-url")
async def analyze_image_mix_boolean_url(
    parameter_choice: str,
    image_url: str,
):
    """
    Analyze parameters in an image from a URL
    
    Args:
        image_url: URL of the image to analyze
        parameter_choice: Comma-separated indices of parameters to analyze
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Leading Space
    2. Leading Lines
    3. Pattern Recognition
    4. Retouching
    5. Symmetrical Balance
    6. Frame in a frame
    7. Haze Presence
    8. Diagonal Leading Lines
    9. Double Exposure
    10. HDR
    11. Lens Flare

    """
    try:
        # Download image from URL
                
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
        # Verify it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL must point to an image")
        
        input_values = parameter_choice.split(',') 
        parameters_list = [int(s.strip()) for s in input_values]
        

        chosen_params = ''
        for param in parameters_list:
            if chosen_params == "":
                chosen_params = my_configs.get_boolean_lora_params()[param-1]
            else:
                chosen_params = chosen_params + f', {my_configs.get_boolean_lora_params()[param-1]}'
        instruction = f"Analyze the image for {chosen_params}."
        model = my_configs.get_boolean_lora_path()
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        # Extract filename from URL for logging and saving
        filename = image_url.split('/')[-1].split('?')[0] or "url_image.jpg"
        # Ensure filename has an extension
        if '.' not in filename:
            filename += '.jpg'
        

        logger.info(f"Processing image from URL: {image_url} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {chosen_params}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # Save image to SAVE_DIR
        # save_path = os.path.join(SAVE_DIR, filename)
        # with open(save_path, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": filename
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = chosen_params
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except httpx.HTTPError as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/analyze-image-mix-scoring")
async def analyze_image_mix_scoring(
    parameter_choice: str,
    file: UploadFile = File(...),
):
    """
    Analyze parameters in an uploaded image
    
    Args:
        file: Uploaded image file
        instruction: Custom instruction text for the analysis
        model_choice: Model to use (base, lora, or merged)
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Exposure
    2. Perspective Shift
    3. Perspective Lines
    4. Light Softness Viewer
    5. Light Reflection
    6. Pattern Repetition
    7. Point of Light
    8. Sense of Motion
    9. Dust Visibility
    10. Soft Focus\n
    11. Digital Noise
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        input_values = parameter_choice.split(',') 
        parameters_list = [int(s.strip()) for s in input_values]
        

        chosen_params = ''
        for param in parameters_list:
            if chosen_params == "":
                chosen_params = my_configs.get_scoring_lora_params()[param-1]
            else:
                chosen_params = chosen_params + f', {my_configs.get_scoring_lora_params()[param-1]}'
        instruction = f"Analyze the image for {chosen_params}."
        model = my_configs.get_scoring_lora_path()
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        

        logger.info(f"Processing im`age: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {chosen_params}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": file.filename or "uploaded_image"
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = chosen_params
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
    
@app.post("/analyze-image-mix-scoring-url")
async def analyze_image_mix_scoring_url(
    parameter_choice: str,
    image_url: str,
):
    """
    Analyze parameters in an image from a URL
    
    Args:
        image_url: URL of the image to analyze
        parameter_choice: Comma-separated indices of parameters to analyze
        
    Returns:
        JSON response with single analysis
        
        
    You should only enter the index of the parameter you are looking for:
    1. Exposure
    2. Perspective Shift
    3. Perspective Lines
    4. Light Softness Viewer
    5. Light Reflection
    6. Pattern Repetition
    7. Point of Light
    8. Sense of Motion
    9. Dust Visibility
    10. Soft Focus
    11. Digital Noise
    """
    try:
        # Download image from URL
        
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
        # Verify it's an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="URL must point to an image")
        
        input_values = parameter_choice.split(',') 
        parameters_list = [int(s.strip()) for s in input_values]
        

        chosen_params = ''
        for param in parameters_list:
            if chosen_params == "":
                chosen_params = my_configs.get_scoring_lora_params()[param-1]
            else:
                chosen_params = chosen_params + f', {my_configs.get_scoring_lora_params()[param-1]}'
        instruction = f"Analyze the image for {chosen_params}."
        model = my_configs.get_scoring_lora_path()
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        # Extract filename from URL for logging and saving
        filename = image_url.split('/')[-1].split('?')[0] or "url_image.jpg"
        # Ensure filename has an extension
        if '.' not in filename:
            filename += '.jpg'
        

        logger.info(f"Processing image from URL: {image_url} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {chosen_params}")
        logger.info(f"Selected model: {model}")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # Save image to SAVE_DIR
        # save_path = os.path.join(SAVE_DIR, filename)
        # with open(save_path, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                        "uuid": filename
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content

        logger.info(f"Received response from vLLM server")
        logger.info(f"Response was:\n{response_content}")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            parameter_value = chosen_params
            full_response = {'Parameter': parameter_value, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "image_url": image_url,
                "model_used": model,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except httpx.HTTPError as e:
        logger.error(f"Error downloading image from URL: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    

@app.get("/health")
async def health_check():
    """Health check endpoint that also tests vLLM server connectivity"""
    try:
        return {
            "status": "healthy",
            "vllm_server": openai_api_base,
            "message": "API is running and ready to process images"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "message": "Cannot connect to vLLM server"
            }
        )
        
# Add this diagnostic endpoint to your FastAPI app
@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        # Try to get model config from vLLM
        model_path = my_configs.get_base_model_path()
        
        # Check if client has model info method
        if hasattr(client, 'models'):
            models = client.models.list()
            return {"models": [m.id for m in models.data]}
        
        return {
            "model_path": model_path,
            "note": "Query vLLM server directly for detailed info"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8005,
        log_level="info",
        access_log=True
    )
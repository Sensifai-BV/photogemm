from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import base64
import io
import time
import os
import httpx
import asyncio
import argparse
from PIL import Image
from openai import OpenAI
import json
import logging
from enum import Enum
from handle_non_challenging_request import NonChallegingHandler
from parameters_naming import ModelConfig
from preprcessors import GoldenRectangle, RuleOfThird


class ModelType(str, Enum):
    PHOTOGEMM = "photogemm"  # PhotoGemm model on port 8000 - Multiple specialized endpoints
    PHOTOGEMMQ = "photogemmq"  # PhotoGemmQ model on port 8080 - Single quantized endpoint

# This will be set from command line arguments
CURRENT_MODEL_TYPE = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Image Analysis API - PhotoGemm/PhotoGemmQ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Run with PhotoGemm (default)
  python app.py --is_quantized 0   # Run with PhotoGemm (explicit)
  python app.py --is_quantized 1   # Run with PhotoGemmQ (quantized)
        """
    )
    parser.add_argument(
        '--is_quantized',
        type=int,
        choices=[0, 1],
        default=0,
        help='Use quantized model: 0 = PhotoGemm (default), 1 = PhotoGemmQ'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8005,
        help='Port to run the API server on (default: 8005)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to run the API server on (default: 0.0.0.0)'
    )
    return parser.parse_args()

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
class ModelEndpointConfig:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        
        if model_type == ModelType.PHOTOGEMMQ:
            self.api_base = "http://0.0.0.0:8080/v1"
            self.model_name = "merged-model-vllm"
            self.description = "PhotoGemmQ - Quantized single model endpoint"
        else:  # PHOTOGEMM
            self.api_base = "http://0.0.0.0:8000/v1"
            self.model_name = None  # Will be determined per request
            self.description = "PhotoGemm - Multiple specialized model endpoints"
    
    def get_client(self):
        return OpenAI(
            api_key="EMPTY",
            base_url=self.api_base,
        )
    
    def is_photogemm(self):
        """Check if using PhotoGemm (multiple specialized endpoints)"""
        return self.model_type == ModelType.PHOTOGEMM
    
    def is_photogemmq(self):
        """Check if using PhotoGemmQ (single quantized endpoint)"""
        return self.model_type == ModelType.PHOTOGEMMQ
    
    def get_model_for_request(self, specific_model=None):
        if self.model_type == ModelType.PHOTOGEMMQ:
            return self.model_name
        return specific_model if specific_model else "merged-model-vllm"

# ============================================================================
# INITIALIZATION
# ============================================================================
# These will be initialized in main() after parsing arguments
endpoint_config = None
client = None
app = None
non_challenging_handler = None
my_configs = None
rule_of_thirds_preprocessor = None
golden_rectangle_preprocessor = None

def initialize_app(model_type: ModelType):
    """Initialize the FastAPI app and all components with the specified model type"""
    global endpoint_config, client, app, non_challenging_handler, my_configs
    global rule_of_thirds_preprocessor, golden_rectangle_preprocessor, CURRENT_MODEL_TYPE
    
    CURRENT_MODEL_TYPE = model_type
    
    endpoint_config = ModelEndpointConfig(CURRENT_MODEL_TYPE)
    client = endpoint_config.get_client()
    
    app = FastAPI(
        title="Image Analysis API",
        version="2.0.0",
        description=f"Running in {CURRENT_MODEL_TYPE.value} model mode - {endpoint_config.description}"
    )
    
    non_challenging_handler = NonChallegingHandler()
    my_configs = ModelConfig()
    rule_of_thirds_preprocessor = RuleOfThird()
    golden_rectangle_preprocessor = GoldenRectangle()
    
    logger.info(f"🚀 Initialized API with {CURRENT_MODEL_TYPE.value} model")
    logger.info(f"📍 Endpoint: {endpoint_config.api_base}")
    logger.info(f"📝 Description: {endpoint_config.description}")
    
    # Register all routes after app is initialized
    register_routes()
    
    return app



def image_to_base64_data_url(image: Image.Image, format: str = "JPEG") -> str:
    """Convert PIL Image to base64 data URL"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def check_endpoint_availability(endpoint_type: str):
    """Decorator to check if endpoint is available for current model type"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if endpoint_type == "photogemm" and not endpoint_config.is_photogemm():
                raise HTTPException(
                    status_code=503,
                    detail=f"Endpoint '{func.__name__}' is only available with PhotoGemm model. "
                           f"Current model: {CURRENT_MODEL_TYPE.value}. "
                           f"Switch to PHOTOGEMM mode to use this endpoint."
                )
            elif endpoint_type == "photogemmq" and not endpoint_config.is_photogemmq():
                raise HTTPException(
                    status_code=503,
                    detail=f"Endpoint '{func.__name__}' is only available with PhotoGemmQ model. "
                           f"Current model: {CURRENT_MODEL_TYPE.value}. "
                           f"Switch to PHOTOGEMMQ mode to use this endpoint."
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator



# ENDPOINTS



def register_routes():
    """Register all API routes - called after app initialization"""
    
    @app.get("/")
    async def root():
        """Health check endpoint"""
        return {
            "message": "Image Analysis API is running",
            "model": CURRENT_MODEL_TYPE.value,
            "description": endpoint_config.description,
            "endpoint": endpoint_config.api_base,
            "available_endpoints": get_available_endpoints()
        }

def get_available_endpoints():
    """Return list of available endpoints based on model type"""
    base_endpoints = [
        "/health",
        "/model-info",
        "/config"
    ]
    
    # PhotoGemmQ endpoints (optimized single endpoint)
    photogemmq_endpoints = [
        "/analyze-image-single"
    ]
    
    # PhotoGemm endpoints (multiple specialized endpoints)
    photogemm_endpoints = [
        "/analyze-image-single",
        "/analyze-image-multiple",
        "/analyze-image-url",
        "/analyze-image-mix-boolean",
        "/analyze-image-mix-boolean-url",
        "/analyze-image-mix-scoring",
        "/analyze-image-mix-scoring-url"
    ]
    
    if endpoint_config.is_photogemmq():
        return base_endpoints + photogemmq_endpoints
    else:  # PhotoGemm
        return base_endpoints + photogemm_endpoints

@app.get("/config")
async def get_config():
    """Get current configuration"""
    return {
        "model": CURRENT_MODEL_TYPE.value,
        "description": endpoint_config.description,
        "api_base": endpoint_config.api_base,
        "is_photogemm": endpoint_config.is_photogemm(),
        "is_photogemmq": endpoint_config.is_photogemmq(),
        "available_endpoints": get_available_endpoints()
    }



# SINGLE IMAGE ANALYSIS (Available in both PhotoGemm and PhotoGemmQ)

@app.post("/analyze-image-single")
async def analyze_image_single(
    parameter_choices: int,
    image_file: UploadFile = File(...),
):
    """
    Analyze parameters in an uploaded image
    Available in both PhotoGemm and PhotoGemmQ modes
    
    PhotoGemm: Uses specialized models for different parameters
    PhotoGemmQ: Uses single quantized merged model for all parameters
    
    Parameters: 1-59 (see full list in documentation)
    """
    try:
        parameter_choice = parameter_choices
        file = image_file
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        
        # Process based on parameter choice
        if parameter_choice == 12:
            processed_image = rule_of_thirds_preprocessor.process(data_input=Image.open(io.BytesIO(image_data)))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG") 
            image_data = byte_buffer.getvalue()
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            model = my_configs.get_rot_model_path() if endpoint_config.is_photogemm() else None
        
        elif parameter_choice == 59:
            processed_image = golden_rectangle_preprocessor.process(data_input=Image.open(io.BytesIO(image_data)))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG")
            image_data = byte_buffer.getvalue()
            choosen_param = "Golden Rectangle"
            instruction = f"Analyze the image for the 'golden rectangle' parameter and provide the result as a JSON object."
            if hasattr(non_challenging_handler, 'ending_prompt_bool'):
                instruction = instruction + non_challenging_handler.ending_prompt_bool
            model = my_configs.get_gr_model_path() if endpoint_config.is_photogemm() else None
            
        elif parameter_choice < 36 or parameter_choice == 46 or parameter_choice == 42:
            if parameter_choice == 46: parameter_choice = 37
            if parameter_choice == 42: parameter_choice = 38
            
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            model = my_configs.get_base_model_path() if endpoint_config.is_photogemm() else None
        
        elif parameter_choice > 35 and parameter_choice < 47:
            choosen_param = my_configs.get_normal_bool_names()[parameter_choice-36]
            instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
            if hasattr(non_challenging_handler, 'ending_prompt_bool'):
                instruction = instruction + non_challenging_handler.ending_prompt_bool
            model = my_configs.get_boolean_lora_path() if endpoint_config.is_photogemm() else None
            
        else:
            if parameter_choice == 58:
                choosen_param = "Hue"
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                if hasattr(non_challenging_handler, 'ending_prompt_others'):
                    instruction = instruction + non_challenging_handler.ending_prompt_others
                model = my_configs.get_lora_model_path() if endpoint_config.is_photogemm() else None
            else:
                choosen_param = my_configs.get_normal_scoring_names()[parameter_choice-47]
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                if hasattr(non_challenging_handler, 'ending_prompt_score'):
                    instruction = instruction + non_challenging_handler.ending_prompt_score
                model = my_configs.get_scoring_lora_path() if endpoint_config.is_photogemm() else None
        
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        logger.info(f"Processing image: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {choosen_param}")
        logger.info(f"Model: {CURRENT_MODEL_TYPE.value}")
        
        # Determine which model to use
        model_to_use = endpoint_config.get_model_for_request(model)
        
        chat_response = client.chat.completions.create(
            model=model_to_use,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }],
            temperature=0 
        )
        
        response_content = chat_response.choices[0].message.content
        logger.info(f"Received response from vLLM server")
        
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content
            
            parsed_response = json.loads(json_str)
            full_response = {'Parameter': choosen_param, **parsed_response}
            
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model_to_use,
                "model": CURRENT_MODEL_TYPE.value,
                "analysis": full_response,
                "raw_response": response_content
            })
            
        except json.JSONDecodeError:
            return JSONResponse(content={
                "success": True,
                "filename": file.filename,
                "model_used": model_to_use,
                "raw_response": response_content,
                "note": "Response could not be parsed as JSON"
            })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/analyze-image-multiple")
@check_endpoint_availability("photogemm")
async def analyze_image_multiple(
    parameter_choices: str,
    file: UploadFile = File(...),
):
    """
    Analyze multiple parameters in an uploaded image asynchronously
    
    **PhotoGemm ONLY** - Uses specialized models with batch processing
    Not available in PhotoGemmQ mode
    
    Args:
        parameter_choices: Comma-separated string of parameter indices
        file: Uploaded image file
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        try:
            param_list = [int(p.strip()) for p in parameter_choices.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid parameter format")
        
        if any(p < 1 or p > 59 for p in param_list):
            raise HTTPException(status_code=400, detail="Parameters must be between 1 and 59")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        
        logger.info(f"Processing image: {file.filename} with parameters: {param_list}")
        
        # Group parameters by model
        base_params = []
        boolean_params = []
        scoring_params = []
        hue_params = []
        
        for param in param_list:
            if param < 36:
                base_params.append(param)
            elif 36 <= param < 47:
                boolean_params.append(param)
            elif param == 58:
                hue_params.append(param)
            else:
                scoring_params.append(param)
        
        async def call_base_model_batch(params_batch):
            model_path = my_configs.get_base_model_path()
            instructions = []
            param_names = []
            
            for param in params_batch:
                instruction, param_name = non_challenging_handler(param)
                instructions.append(instruction)
                param_names.append(param_name)
            
            merged_instruction = "\n".join(instructions)
            merged_instruction += f"\n\nReturn the response for the {len(params_batch)} given parameters in a single dictionary."
            
            chat_response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": merged_instruction},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }],
                temperature=0
            )
            
            response_content = chat_response.choices[0].message.content
            
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
                    "analysis": parsed_response,
                    "raw_response": response_content
                }
            except json.JSONDecodeError:
                return {
                    "group": "base_model_batch",
                    "model": model_path,
                    "parameters": param_names,
                    "raw_response": response_content,
                    "note": "Response could not be parsed as JSON"
                }
        
        async def call_batched_model(params, model_path, group_name, name_getter):
            if not params:
                return None
            
            param_names = [name_getter(param) for param in params]
            param_list_str = ", ".join(param_names)
            instruction = f"Analyze the image for the following parameters: {param_list_str}."
            
            chat_response = client.chat.completions.create(
                model=model_path,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }],
                temperature=0
            )
            
            response_content = chat_response.choices[0].message.content
            
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
                    "analysis": parsed_response,
                    "raw_response": response_content
                }
            except json.JSONDecodeError:
                return {
                    "group": group_name,
                    "model": model_path,
                    "parameters": param_names,
                    "raw_response": response_content,
                    "note": "Response could not be parsed as JSON"
                }
        
        tasks = []
        base_batch_size = 18
        
        if base_params:
            for i in range(0, len(base_params), base_batch_size):
                batch = base_params[i:i + base_batch_size]
                tasks.append(call_base_model_batch(batch))
        
        if boolean_params:
            tasks.append(call_batched_model(
                boolean_params,
                my_configs.get_boolean_lora_path(),
                "boolean_lora",
                lambda p: my_configs.get_normal_bool_names()[p-36]
            ))
        
        if scoring_params:
            tasks.append(call_batched_model(
                scoring_params,
                my_configs.get_scoring_lora_path(),
                "scoring_lora",
                lambda p: my_configs.get_normal_scoring_names()[p-47]
            ))
        
        if hue_params:
            tasks.append(call_batched_model(
                hue_params,
                my_configs.get_lora_model_path(),
                "hue_lora",
                lambda p: "Hue"
            ))
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed_time = time.perf_counter() - start_time
        
        successful_results = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif result is not None:
                successful_results.append(result)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "model": CURRENT_MODEL_TYPE.value,
            "requested_parameters": param_list,
            "results": successful_results,
            "errors": errors if errors else None,
            "execution_time_sec": round(elapsed_time, 2)
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")



@app.post("/analyze-image-url")
@check_endpoint_availability("photogemm")
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



@app.post("/analyze-image-mix-boolean")
@check_endpoint_availability("photogemm")
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


@app.post("/analyze-image-mix-scoring")
@check_endpoint_availability("photogemm")
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
    




@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "model": CURRENT_MODEL_TYPE.value,
            "description": endpoint_config.description,
            "vllm_server": endpoint_config.api_base,
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

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    try:
        return {
            "model": CURRENT_MODEL_TYPE.value,
            "description": endpoint_config.description,
            "api_base": endpoint_config.api_base,
            "is_photogemm": endpoint_config.is_photogemm(),
            "is_photogemmq": endpoint_config.is_photogemmq(),
            "available_endpoints": get_available_endpoints(),
            "capabilities": {
                "photogemm": "Multiple specialized models with batch processing",
                "photogemmq": "Single quantized merged model for optimized performance"
            }
        }
    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Determine model type based on --is_quantized flag
    if args.is_quantized == 1:
        model_type = ModelType.PHOTOGEMMQ
        logger.info("🔧 Using PhotoGemmQ (quantized model)")
    else:
        model_type = ModelType.PHOTOGEMM
        logger.info("🔧 Using PhotoGemm (multiple specialized models)")
    
    # Initialize the app with selected model type
    app = initialize_app(model_type)
    
    # Print startup information
    print("\n" + "="*70)
    print(f"🚀 Starting Image Analysis API")
    print("="*70)
    print(f"Model Type:    {model_type.value}")
    print(f"Description:   {endpoint_config.description}")
    print(f"API Endpoint:  {endpoint_config.api_base}")
    print(f"Server Host:   {args.host}")
    print(f"Server Port:   {args.port}")
    print(f"Docs URL:      http://{args.host}:{args.port}/docs")
    print("="*70 + "\n")
    
    # Run the server
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info",
        access_log=True
    )
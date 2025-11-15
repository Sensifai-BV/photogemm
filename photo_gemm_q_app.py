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
openai_api_base = "http://0.0.0.0:8080/v1"

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
    parameter_choices: int,
    image_file: UploadFile = File(...),
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
        parameter_choice = parameter_choices
        file = image_file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        
        if parameter_choice == 12:
            processed_image = rule_of_thirds_preprocessor.process(data_input=Image.open(file.file))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG") 
            image_data = byte_buffer.getvalue()
            instruction, choosen_param = non_challenging_handler(parameter_choice)
            model = my_configs.get_rot_model_path()
        
        elif parameter_choice == 59:
            processed_image = golden_rectangle_preprocessor.process(data_input=Image.open(file.file))
            byte_buffer = io.BytesIO()
            processed_image.save(byte_buffer, format="PNG")
            image_data = byte_buffer.getvalue()
            choosen_param = "Golden Rectangle"
            instruction = f"Analyze the image for the 'golden rectangle' parameter and provide the result as a JSON object."
            instruction = instruction + non_challenging_handler.ending_prompt_bool
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
            instruction = instruction + non_challenging_handler.ending_prompt_bool
            model = my_configs.get_boolean_lora_path()
            
        else:
            if parameter_choice == 58:
                choosen_param = "Hue"
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                instruction = instruction + non_challenging_handler.ending_prompt_others
                model = my_configs.get_lora_model_path()
            else:
                choosen_param = my_configs.get_normal_scoring_names()[parameter_choice-47]
                instruction = f"Analyze the image for the {choosen_param} parameter and provide the result as a JSON object."
                instruction = instruction + non_challenging_handler.ending_prompt_score
                model = my_configs.get_scoring_lora_path()
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_data_url = image_to_base64_data_url(image)
        

        logger.info(f"Processing im`age: {file.filename} (size: {len(image_data)} bytes)")
        logger.info(f"Selected param: {choosen_param}")
        logger.info(f"Selected model: merged-model-vllm")
        logger.info(f"Instruction: {instruction[:100]}...")  # Log first 100 chars
        
        # save_path = os.path.join(SAVE_DIR, file.filename)
        # with open(file.filename, "wb") as f:
        #     f.write(image_data)
            
        print(instruction)

        chat_response = client.chat.completions.create(
            model="merged-model-vllm",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
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
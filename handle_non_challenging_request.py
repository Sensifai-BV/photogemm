import json


PARAMETERS_PATH = "/home/ubuntu/parsa/vllm_inference/test002/filtered_parameters.json"
NON_CHALLENGING_PARAMS = [
"Minimizing Distractions",
"Subtle Complexity",
"Composition Contrast",
"Depth of Field",
"Negative Space",
"Texture",
"Aspect Ratio",
"Implied Lines",
"Scale and Proportion",
"Color Harmony",
"Location Setting",
"Rule of Thirds",
"Haze Impact",
"Dust Existence",
"Fringe",
"Starburst",
"Moire",
"Luminosity",
"Saturation",
"Contrast",
"Sharpness",
"Light Source Viewer",
"Color Temperature",
"Visual Harmony",
"Overall Expert Assessment",
"Overall Technical Assessment",
"Development",
"Image Realism",
"AI Creation Probabiltiy",
"Photographic Intent",
"Emotional Impact",
"Intended Purpose",
"Target Audience",
"Detailed Emotional Impact",
"ColorMode",
]


class NonChallegingHandler():
    def __init__(self):
        with open(PARAMETERS_PATH, 'r') as f:
            self.prompt_dictionary = json.load(f)       
    
        self.starting_prompt = "Analyze the given photo for the $$$$ parameter.\n"
        self.ending_prompt_score = "\nAssign a flexible score between 0–100 that aligns logically with your explanation.\nReturn the output as a markdown JSON snippet with the keys score and explanation nested under a main Result key."
        self.ending_prompt_bool = "\nAssign a boolean score that aligns logically with your explanation.\nReturn the output as a markdown JSON snippet with the keys score and explanation nested under a main Result key."
        self.ending_prompt_text = "\nReturn the output as a markdown JSON snippet with an explanation key as a main Result key."
        self.ending_prompt_others = "\nReturn the output as a markdown JSON snippet with the keys score and explanation nested under a main Result key."
    
    def __call__(self, parameter_index: int):
        parameter_name = NON_CHALLENGING_PARAMS[parameter_index-1]
        starting_prompt = self.starting_prompt.replace('$$$$', parameter_name)
        finded_param = {}
        for record in self.prompt_dictionary:
            if record['Parameter'].lower() == parameter_name.lower():
                finded_param = record
                break
        
        print("***")
        print(finded_param)    
        print("***")
        
        if finded_param['New Prompt'].startswith("#"):
            final_prompt = finded_param['New Prompt']
        
        elif finded_param['Evaluation'] == "Boolean":
            final_prompt = starting_prompt + finded_param['New Prompt'] + self.ending_prompt_bool
            
        elif finded_param['Evaluation'] == "Score":
            final_prompt = starting_prompt + finded_param['New Prompt'] + self.ending_prompt_score

        elif finded_param['Evaluation'] == "Text":
            final_prompt = starting_prompt + finded_param['New Prompt'] + self.ending_prompt_text
            
        else:
            final_prompt = starting_prompt + finded_param['New Prompt'] + self.ending_prompt_others
        
        
        return final_prompt, parameter_name
class ModelConfig:
    def __init__(self):
        self.base_model_path = "/home/ubuntu/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
        self.lora_model_path = "my_gemma_lora"
        self.boolean_lora_path = "my_lora_boolean"
        self.scoring_model_path = "my_scoring_lora"
        
        self.normnal_lora_names = [
            'Soft Focus',
            'Hue',
            'Dust Visibility',
            'Point of Light',
        ]
        
        self.lora_params = [
            'soft_focus',
            'hue',
            'dust_visibility',
            'point_of_light',
        ]
        
        self.normal_bool_names = [
            'Leading Space',
            'Leading Lines',
            'Pattern Recognition',
            'Retouching',
            'Symmetrical Balance',
            'Frame in frame',
            'Haze Presence',
            'Diagonal Leading Lines',
            #'ColorMode',
            'Double Exposure',
            'HDR',
            'Lens Flare',
        ]
        
        self.boolean_lora_params = [
            'leading_space',
            'leading_lines',
            'pattern_recognition',
            'retouching',
            'symmetrical_balance',
            'frame_in_frame',
            'haze_presence',
            'diagonal_leading_lines',
            #'colormode',
            'double_exposure',
            'hdr',
            'lens_flare',
        ]
        
        self.normal_scoring_names = [
            'Exposure',
            'Perspective Shift',
            'Perspective Lines',
            'Light Softness Viewer',
            'Light Reflection',
            'Pattern Repetition',
            'Point of Light',
            'Sense of Motion',
            'Dust Visibility',
            'Soft Focus',
            'Digital noise',
        ]
        
        self.scoring_lora_params = [
            'exposure',
            'perspective_shift',
            'perspective_lines',
            'light_softness_viewer',
            'light_reflection',
            'pattern_repetition',
            'point_of_light',
            'sense_of_motion',
            'dust_visibility',
            'soft_focus',
            'digital_noise',
        ]



    def get_base_model_path(self):
        return self.base_model_path

    def get_lora_model_path(self):
        return self.lora_model_path

    def get_boolean_lora_path(self):
        return self.boolean_lora_path

    def get_normal_lora_names(self):
        return self.normnal_lora_names

    def get_lora_params(self):
        return self.lora_params

    def get_normal_bool_names(self):
        return self.normal_bool_names

    def get_boolean_lora_params(self):
        return self.boolean_lora_params
    
    def get_normal_scoring_names(self):
        return self.normal_scoring_names

    def get_scoring_lora_path(self):
        return self.scoring_model_path
    
    def get_scoring_lora_params(self):
        return self.scoring_lora_params
    
# # --- Example Usage ---
# if __name__ == "__main__":
#     config = ModelConfig()
    
#     # You can now get each value using its function
#     print(f"Base Model: {config.get_base_model_path()}")
#     print(f"LoRA Params: {config.get_lora_params()}")
#     print(f"Boolean Names: {config.get_normal_bool_names()}")
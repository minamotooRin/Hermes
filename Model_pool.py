from Models.LLaMA3 import LLaMA3
from Models.DeepSeek import DeepSeek
from Models.Qwen import Qwen
from Models.Gemini import Gemini
from Models.LLaVA import LLaVA

class ModelPool:
    def __init__(self):

        self.type2class = {
            "LLaMA3": LLaMA3,
            "DeepSeek": DeepSeek,
            "Qwen": Qwen,
            "Gemini": Gemini,
            "LLaVA": LLaVA,
        }

        self.models = {}
    
    def __getitem__(self, model_name):
        return self.models[model_name]

    def add_local_model(self, model_type, model_name, device, bf16 = True):
        if model_name in self.models:
            return
        self.models[model_name] = self.type2class[model_type](model_name, device, bf16)

    def add_remote_model(self, model_type, model_name, api_key):
        if model_name in self.models:
            return
        self.models[model_name] = self.type2class[model_type](model_name, api_key)
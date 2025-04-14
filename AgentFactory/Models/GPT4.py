# TBD

from AgentFactory.Models.LLM import LLM_remote

class GPT(LLM_remote):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name, api_key, instr = "You are a helpful assistant."):
        super().__init__(model_name, api_key)

        self.roleNames = {
            "system": "model",
            "user": "user",
            "assistant": "model"
        }

        self.instr = instr

        self.PRICE = {
        }

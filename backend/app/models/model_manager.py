import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_REGISTRY = {
    "distilgpt2": {
        "base_model": "distilgpt2",
        "adapter_repo": "hammad051/slm-qna-distilgpt2"
    },
    "qwen2.5-0.5b": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_repo": "hammad051/slm-qna-qwen2.5-0.5b"
    },
    "tinyllama": {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter_repo": "hammad051/slm-qna-tinyllama"
    }
}

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.current_model_id = None

    def _unload_current_model(self):
        if self.model is not None:
            logger.info(f"Unloading model '{self.current_model_id}' to free up GPU memory...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def get_model(self, model_id: str):
        if model_id not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_id}' is not supported. Available models: {list(MODEL_REGISTRY.keys())}")
            
        # Return from cache if it's already the loaded model
        if self.current_model_id == model_id and self.model is not None:
            logger.info(f"Using cached model '{model_id}' from memory.")
            return self.model, self.tokenizer, self.device

        # Unload the previous model to avoid OOM errors
        if self.current_model_id is not None and self.current_model_id != model_id:
            self._unload_current_model()

        # Lazy import of ML dependencies
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
        except ImportError as e:
            logger.error(f"ML dependencies not available: {e}")
            raise RuntimeError("Inference environment is not available. ML dependencies (torch, transformers, peft) are not installed.") from e

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        registry_info = MODEL_REGISTRY[model_id]
        base_model_id = registry_info["base_model"]
        adapter_repo = registry_info["adapter_repo"]

        try:
            logger.info(f"Loading {base_model_id} base model and tokenizer for {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
            
            logger.info(f"Loading LoRA adapter from Hugging Face: {adapter_repo}...")
            self.model = PeftModel.from_pretrained(base_model, adapter_repo)
            
            logger.info(f"Moving model to {self.device}...")
            self.model.to(self.device)
            self.model.eval()
            
            self.current_model_id = model_id
            logger.info(f"{model_id} loaded successfully and ready for inference.")
            
            return self.model, self.tokenizer, self.device
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            self.model = None
            self.tokenizer = None
            self.current_model_id = None
            raise RuntimeError(f"Failed to load model {model_id}: {e}") from e

# Global instance to avoid reloading
model_manager = ModelManager()

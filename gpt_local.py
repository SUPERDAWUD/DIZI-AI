import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    'all': 'tiiuae/falcon-7b-instruct',                 # All-around open model
    'performance': 'EleutherAI/gpt-neox-20b',           # Largest open model for quality
    'fast': 'bigcode/starcoder2-3b',                    # Fastest open model
    'code': 'bigcode/starcoder2-3b',                    # Best open code model
    'math': 'EleutherAI/gpt-neo-2.7B',                  # Good open math/reasoning model
    'chat': 'tiiuae/falcon-7b-instruct',                # Open chat model
    'bias': 'tiiuae/falcon-7b-instruct',                # Use prompt injection for bias
    'oss': 'EleutherAI/gpt-neox-20b',                   # Best open-source SOTA
    'super': 'tiiuae/falcon-7b-instruct',               # Super model orchestrator uses this as local base
}

class LocalGPT:
    def __init__(self, model_name='all', personality='friendly', bias_prompt=None, device='auto'):
        self.model_key = model_name
        self.personality = personality
        self.bias_prompt = bias_prompt
        # Allow overriding device via env LOCAL_DEVICE or FORCE_GPU
        env_dev = os.environ.get('LOCAL_DEVICE', '').strip().lower()
        if env_dev in ('cuda','gpu','cpu'):
            device = 'cuda' if env_dev in ('cuda','gpu') else 'cpu'
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self._try_load_model(MODEL_REGISTRY[model_name])

    def _try_load_model(self, repo_id):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
            force_gpu = os.environ.get('FORCE_GPU', 'false').strip().lower() in ('1','true','yes','on')
            use_cuda = (self.device == 'cuda' and torch.cuda.is_available())
            load_kwargs = {"low_cpu_mem_usage": True}
            if use_cuda and force_gpu:
                # Load model fully on GPU in fp16 and keep all shards on CUDA
                # device_map can be a mapping to force single-GPU placement
                load_kwargs.update({
                    "device_map": {"": self.device},
                    "dtype": torch.float16,
                })
            else:
                # Default: let HF/accelerate place layers; try 8/4-bit if available
                load_kwargs.update({
                    "device_map": "auto",
                    "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                })
                if torch.cuda.is_available():
                    try:
                        import bitsandbytes  # noqa: F401
                        quant = os.environ.get('QUANTIZATION', '8bit').lower()
                        if quant == '4bit':
                            load_kwargs["load_in_4bit"] = True
                        else:
                            load_kwargs["load_in_8bit"] = True
                    except Exception:
                        pass
            self.model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if use_cuda:
                try:
                    self.model.to(self.device)
                except Exception:
                    # If quantized with device_map=auto, .to might be a no-op
                    pass
            self.model.eval()
        except Exception as e:
            # Fallback to echo mode if downloading fails
            print(f"[LocalGPT] Warning: failed to load '{repo_id}': {e}. Using fallback.")

    def set_model(self, model_name, personality='friendly', bias_prompt=None):
        if model_name != self.model_key:
            self._try_load_model(MODEL_REGISTRY[model_name])
            self.model_key = model_name
        self.personality = personality
        self.bias_prompt = bias_prompt

    def set_device(self, device):
        self.device = self._resolve_device(device)
        if self.model is not None:
            self.model.to(self.device)

    def _resolve_device(self, device):
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def generate(self, prompt, max_length=180, do_sample=False, temperature=0.7, top_p=0.95, top_k=50, num_return_sequences=1):
        persona = {
            'friendly': 'You are a friendly, helpful AI assistant.',
            'formal': 'You are a formal, precise AI assistant.',
            'sarcastic': 'You are a witty, sarcastic AI assistant.',
            'creative': 'You are a creative, imaginative AI assistant.',
            'direct': 'You are a direct, concise AI assistant.'
        }.get(self.personality, 'You are a helpful AI assistant.')
        full_prompt = f"{persona}\n"
        if self.bias_prompt:
            full_prompt += f"{self.bias_prompt}\n"
        full_prompt += prompt
        if self.model is None or self.tokenizer is None:
            return full_prompt  # Fallback echo
        inputs = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)
        gen_kwargs = {
            "max_length": max_length,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample or num_return_sequences > 1:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "num_return_sequences": int(num_return_sequences),
            })
        with torch.no_grad():
            outputs = self.model.generate(inputs, **gen_kwargs)
        if outputs.dim() == 1:
            return self.tokenizer.decode(outputs, skip_special_tokens=True)
        texts = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
        if num_return_sequences == 1:
            return texts[0]
        return texts

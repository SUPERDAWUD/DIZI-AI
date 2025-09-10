# CustomModel

This folder contains your custom AI model for DIZI.

## Files
- `model.py`: PyTorch model definition, loader, and inference function.
- `__init__.py`: Package init.
- `train_custom_llm.py`: Sample training script that accepts chat logs,
  code snippets, or images in `user_chats/` and creates the folder if missing.
- Add additional training utilities and weights here.

## Integration
- Import and use `CustomModel` in your main backend (e.g., `chat.py`).
- Add your model to the model registry and selection logic.
- Ensure output format matches other LLMs for seamless chat integration.

## Next Steps
- Train your model and save weights in this folder.
- Update backend to support switching to this model.

## GPU/CPU
- All loaders and the training script automatically use a CUDA GPU when available
  and fall back to CPU otherwise. Install a CUDA-enabled PyTorch build to take
  advantage of GPU acceleration.

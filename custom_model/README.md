# CustomModel

This folder contains your custom AI model for DIZI.

## Files
- `model.py`: PyTorch model definition, loader, and inference function.
- `__init__.py`: Package init.
- Add training scripts, weights, and utilities here.

## Integration
- Import and use `CustomModel` in your main backend (e.g., `chat.py`).
- Add your model to the model registry and selection logic.
- Ensure output format matches other LLMs for seamless chat integration.

## Next Steps
- Train your model and save weights in this folder.
- Update backend to support switching to this model.

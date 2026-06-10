# DIZI-AI Enhanced - ChatGPT-Like Local AI System

## 🚀 The Most Powerful Local AI Model

DIZI-AI Enhanced is a comprehensive local AI system that provides ChatGPT-like features while running entirely on your machine. It includes automatic training, conversation memory, context awareness, and advanced AI capabilities.

## ✨ Key Features

### 🤖 **Automatic Training System**
- **Self-Improving AI**: Automatically trains on your conversations
- **LoRA Fine-tuning**: Uses efficient LoRA adapters for personalized training
- **Background Training**: Trains every 6 hours (configurable)
- **Smart Data Collection**: Learns from your chat history and uploaded files

### 🧠 **ChatGPT-Like Features**
- **Conversation Memory**: Remembers past conversations for context
- **Smart Follow-ups**: Suggests relevant follow-up questions
- **Context Awareness**: Uses conversation history for better responses
- **Session Management**: Organizes conversations like ChatGPT
- **Memory Search**: Search through past conversations

### 🎯 **Advanced AI Capabilities**
- **Multi-Modal Processing**: Text, images, audio, and video
- **Code Analysis**: Advanced code review and optimization
- **Scientific Computing**: Mathematical modeling and analysis
- **Creative Generation**: Story writing, poetry, and creative content
- **Data Analysis**: Machine learning and statistical analysis

### ⚡ **Performance Optimization**
- **GPU Acceleration**: Automatic CUDA optimization
- **Model Quantization**: 4-bit and 8-bit quantization support
- **Intelligent Caching**: Response caching for faster replies
- **Memory Management**: Efficient GPU memory usage
- **Parallel Processing**: Multi-threaded operations

### 🔒 **Privacy & Security**
- **100% Local**: No data leaves your machine
- **Encrypted Storage**: Secure conversation storage
- **No Tracking**: Complete privacy protection
- **Offline Operation**: Works without internet

## 🚀 Quick Start

### Option 1: Enhanced Startup (Recommended)
```bash
python start_dizi_enhanced.py
```

### Option 2: Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py
```

## 🌐 Access the System

- **Main Interface**: http://localhost:5000
- **Dev Panel**: http://localhost:5000/dev
- **Training Status**: Check the sidebar for training information

## 🎮 Usage Guide

### Starting a Conversation
1. Enter your name when prompted
2. Type your message in the input box
3. Press Enter or click Send
4. Enjoy ChatGPT-like responses!

### Training System
- **Automatic**: Trains every 6 hours automatically
- **Manual**: Click "Start Training" in the sidebar
- **Status**: Check training status in the sidebar
- **Data**: Uses your chat history and uploaded files

### Memory Features
- **Search Memory**: Click the 🔍 button to search past conversations
- **Context Awareness**: Enable "Context" toggle for better responses
- **Follow-ups**: Get smart suggestions after each response

### Advanced Features
- **Code Mode**: Switch to code mode for programming help
- **Image Mode**: Generate and analyze images
- **Memory Mode**: Search and manage conversation history
- **Streaming**: Real-time response streaming (like ChatGPT)

## 🔧 Configuration

### Environment Variables
```bash
# Training Configuration
AUTO_TRAINING=true                    # Enable automatic training
TRAINING_INTERVAL_HOURS=6            # Training frequency
PREFER_LOCAL_STREAM=true             # Use local streaming

# Performance Settings
LOCAL_DEVICE=auto                    # Device selection (auto/cpu/cuda)
QUANTIZATION=4bit                    # Model quantization (4bit/8bit/full)
FORCE_GPU=true                       # Force GPU usage

# Security
DEV_MODE_PASSWORD=supersecret        # Dev panel password
FLASK_SECRET_KEY=devsecret           # Flask secret key
```

### Training Configuration
- **Base Model**: Qwen/Qwen2.5-7B-Instruct (configurable)
- **Training Data**: `user_chats/` directory
- **Output**: `custom_model/lora_out/`
- **Epochs**: 1.0 (configurable)
- **Batch Size**: 1 (configurable)

## 📁 Directory Structure

```
DIZI-AI/
├── server.py                    # Enhanced main server
├── start_dizi_enhanced.py      # Enhanced startup script
├── chat.py                     # Core AI logic
├── custom_model/               # Training system
│   ├── finetune_lora.py       # LoRA fine-tuning
│   └── lora_out/              # Training outputs
├── user_chats/                # Training data
├── uploads/                   # File uploads
├── vector_index/              # RAG index
├── templates/                 # Web interface
├── static/                    # Static assets
└── requirements.txt           # Dependencies
```

## 🎯 ChatGPT-Like Features

### Conversation Management
- **Session History**: Automatic conversation saving
- **Chat Organization**: Multiple conversation threads
- **Search & Filter**: Find past conversations
- **Export/Import**: Backup and restore chats

### Smart Interactions
- **Context Awareness**: Remembers conversation context
- **Follow-up Suggestions**: Intelligent next questions
- **Response Streaming**: Real-time typing effect
- **Message Editing**: Edit and resend messages

### Advanced UI
- **Dark Theme**: Professional ChatGPT-like interface
- **Responsive Design**: Works on all devices
- **Keyboard Shortcuts**: Efficient navigation
- **Drag & Drop**: Easy file uploads

## 🔬 Training System Details

### Automatic Training Process
1. **Data Collection**: Gathers conversations from `user_chats/`
2. **Preprocessing**: Formats data for training
3. **LoRA Training**: Efficient fine-tuning with LoRA adapters
4. **Model Update**: Integrates new knowledge
5. **Validation**: Tests improved responses

### Training Data Sources
- **Chat History**: Your conversations with DIZI
- **Uploaded Files**: Documents, code, images
- **User Preferences**: Learning from your interactions
- **Feedback**: Continuous improvement

### Performance Monitoring
- **Training Status**: Real-time training progress
- **Memory Usage**: GPU/CPU utilization
- **Response Time**: Performance metrics
- **Quality Metrics**: Response quality tracking

## 🛠️ Advanced Configuration

### Model Selection
```python
# Available models
MODELS = {
    "all": "Qwen/Qwen2.5-7B-Instruct",      # All-around
    "performance": "Qwen/Qwen2.5-14B-Instruct", # Best quality
    "fast": "Qwen/Qwen2.5-3B-Instruct",     # Fastest
    "code": "Qwen/Qwen2.5-7B-Code",         # Code expert
    "math": "Qwen/Qwen2.5-7B-Math",         # Math genius
}
```

### Training Parameters
```python
# Training configuration
TRAINING_CONFIG = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",
    "epochs": 1.0,
    "batch_size": 1,
    "learning_rate": 2e-4,
    "max_steps": 100,
    "quantization": "4bit"
}
```

## 🔍 Troubleshooting

### Common Issues

**Training Not Starting**
- Check if `user_chats/` has data
- Verify GPU availability
- Check training status in sidebar

**Slow Responses**
- Enable GPU acceleration
- Use 4-bit quantization
- Check system resources

**Memory Issues**
- Clear conversation memory
- Restart the server
- Check available RAM/VRAM

### Performance Tips
1. **Use GPU**: Enable CUDA for faster processing
2. **Quantization**: Use 4-bit for memory efficiency
3. **Batch Size**: Adjust based on your hardware
4. **Caching**: Enable response caching
5. **Streaming**: Use streaming for better UX

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 20GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 16GB+
- **GPU**: NVIDIA RTX 3060+ (8GB VRAM)
- **Storage**: 50GB+ SSD
- **OS**: Latest versions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the amazing transformers library
- **OpenAI**: For inspiration from ChatGPT
- **NVIDIA**: For GPU acceleration support
- **Community**: For feedback and contributions

## 🆘 Support

- **Documentation**: Check this README
- **Issues**: Report bugs on GitHub
- **Discussions**: Join our community
- **Email**: Contact us for support

---

**DIZI-AI Enhanced** - The most powerful local AI model that learns from you! 🚀

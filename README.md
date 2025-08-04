# WIBA Models

Machine learning models for argument detection, topic extraction, and stance analysis in the WIBA platform.

## 🏗️ Model Architecture

This repository contains three specialized models for different argument mining tasks:

### 1. **WIBADetect** - Argument Detection
- **Base Model**: Llama-3.2-3B-Instruct with LoRA fine-tuning
- **Task**: Binary classification (Argument/NoArgument)
- **Framework**: HuggingFace Transformers + PEFT
- **Location**: `wibadetect/`

### 2. **WIBAExtract** - Topic Extraction  
- **Base Model**: Llama-3.2-3B-Instruct via vLLM
- **Task**: Extract topics from argumentative text
- **Framework**: DSPy for structured reasoning
- **Location**: `wibaextract/`

### 3. **Stance** - Stance Classification
- **Base Model**: Llama-2-7B with LoRA fine-tuning
- **Task**: 3-class classification (Favor/Against/NoArgument)
- **Framework**: HuggingFace Transformers + PEFT
- **Location**: `stance/`

## 📁 Repository Structure

```
wiba-models/
├── wibadetect/          # Argument detection model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── wibaextract/         # Topic extraction model  
│   ├── program.pkl      # DSPy serialized program
│   ├── metadata.json    # Model metadata
│   └── ...
├── stance/              # Stance classification model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
├── shared/              # Shared utilities
│   ├── configs/         # Model configurations
│   ├── utils/           # Model utilities
│   └── evaluation/      # Evaluation scripts
├── scripts/             # Training and evaluation scripts
└── docs/               # Model documentation
```

## 🚀 Usage

### Prerequisites
- Python 3.11+
- PyTorch with CUDA support
- 16GB+ GPU memory recommended

### Loading Models

#### Argument Detection (WIBADetect)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "./wibadetect/",
    num_labels=2,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./wibadetect/")

# Use for inference
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
```

#### Topic Extraction (WIBAExtract)
```python
import dspy
import pickle

# Configure DSPy with vLLM
dspy.configure(lm=dspy.LM(
    model="openai/meta-llama/Llama-3.2-3B-Instruct",
    api_base="http://localhost:8000/v1"
))

# Load serialized program
with open("./wibaextract/program.pkl", "rb") as f:
    topic_extractor = pickle.load(f)

# Use for inference
result = topic_extractor(text="Your argumentative text here")
print(result.topic)
```

#### Stance Classification
```python
from transformers import LlamaForSequenceClassification, AutoTokenizer

# Load model
model = LlamaForSequenceClassification.from_pretrained(
    "./stance/",
    num_labels=3,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("./stance/")

# Format input with topic
text_with_topic = f"Text: {text}\nTopic: {topic}"
inputs = tokenizer(text_with_topic, return_tensors="pt")
outputs = model(**inputs)
```

## 🔧 Model Configurations

### WIBADetect Configuration
```json
{
  "base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
  "peft_type": "LORA",
  "task_type": "SEQ_CLS",
  "r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.1,
  "target_modules": ["v_proj", "up_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "down_proj"]
}
```

### Stance Model Configuration
```json
{
  "base_model_name_or_path": "NousResearch/Llama-2-7b-hf", 
  "peft_type": "LORA",
  "task_type": "SEQ_CLS",
  "r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "num_labels": 3
}
```

## 📊 Model Performance

| Model | Task | Accuracy | F1-Score | Model Size |
|-------|------|----------|----------|------------|
| WIBADetect | Argument Detection | 85.2% | 0.847 | ~194MB (LoRA) |
| WIBAExtract | Topic Extraction | - | - | ~134KB (DSPy) |
| Stance | Stance Classification | 78.9% | 0.785 | ~80MB (LoRA) |

## 🔄 Model Updates

### Version History
- **v1.0.0**: Initial model release
- **v1.1.0**: Improved LoRA configurations
- **v1.2.0**: DSPy integration for topic extraction

### Updating Models
```bash
# Download latest models
git pull origin main

# Verify model integrity
python scripts/verify_models.py
```

## 🧪 Evaluation

### Running Evaluations
```bash
# Evaluate argument detection
python scripts/evaluate_detection.py --model ./wibadetect/

# Evaluate topic extraction  
python scripts/evaluate_extraction.py --model ./wibaextract/

# Evaluate stance classification
python scripts/evaluate_stance.py --model ./stance/
```

### Benchmarks
- Congressional hearing transcripts
- Academic paper abstracts
- Social media discussions
- News article comments

## 🛠️ Development

### Training New Models
```bash
# Train argument detection model
python scripts/train_detection.py --config configs/wibadetect.yaml

# Train stance classification model
python scripts/train_stance.py --config configs/stance.yaml
```

### Model Optimization
- **Quantization**: 4-bit quantization with BitsAndBytes
- **Memory Optimization**: Gradient checkpointing, DeepSpeed integration
- **Inference Optimization**: TensorRT, ONNX conversion support

## 🤝 Contributing

1. **Model Improvements**: Submit PRs with enhanced model configurations
2. **New Models**: Add new argument mining models following the structure  
3. **Evaluation**: Contribute evaluation scripts and benchmarks
4. **Documentation**: Help improve model documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Related Repositories

- **[wiba-platform](https://github.com/WIBA-ORG/wiba-platform)**: Core API server
- **[wiba-python-client](https://github.com/WIBA-ORG/wiba-python-client)**: Python client library
- **[wiba-docs](https://github.com/WIBA-ORG/wiba-docs)**: Documentation and research papers
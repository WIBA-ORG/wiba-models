"""Shared utilities for loading WIBA models."""

import torch
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    LlamaForSequenceClassification,
    BitsAndBytesConfig
)

logger = logging.getLogger(__name__)


class WIBAModelLoader:
    """Utility class for loading WIBA models with optimized configurations."""
    
    @staticmethod
    def get_quantization_config() -> BitsAndBytesConfig:
        """Get standard quantization configuration for WIBA models."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
    
    @staticmethod
    def load_detection_model(model_path: str, device_map: str = "auto") -> tuple:
        """Load WIBA argument detection model.
        
        Args:
            model_path: Path to the model directory
            device_map: Device mapping strategy
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading detection model from {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                torch_dtype=torch.float16,
                device_map=device_map,
                quantization_config=WIBAModelLoader.get_quantization_config(),
                low_cpu_mem_usage=True
            )
            
            model.eval()
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
            
            logger.info("Detection model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load detection model: {str(e)}")
            raise
    
    @staticmethod
    def load_stance_model(model_path: str, device_map: str = "auto") -> tuple:
        """Load WIBA stance classification model.
        
        Args:
            model_path: Path to the model directory
            device_map: Device mapping strategy
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading stance model from {model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = LlamaForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3,
                torch_dtype=torch.float16,
                device_map=device_map,
                quantization_config=WIBAModelLoader.get_quantization_config(),
                low_cpu_mem_usage=True
            )
            
            model.eval()
            
            logger.info("Stance model loaded successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load stance model: {str(e)}")
            raise
    
    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory after model operations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleaned up")


class ModelManager:
    """Manager class for handling multiple WIBA models."""
    
    def __init__(self, model_base_path: str):
        self.model_base_path = Path(model_base_path)
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
    
    def load_detection_model(self) -> None:
        """Load the argument detection model."""
        model_path = self.model_base_path / "wibadetect"
        model, tokenizer = WIBAModelLoader.load_detection_model(str(model_path))
        self.models["detection"] = model
        self.tokenizers["detection"] = tokenizer
    
    def load_stance_model(self) -> None:
        """Load the stance classification model."""
        model_path = self.model_base_path / "stance"
        model, tokenizer = WIBAModelLoader.load_stance_model(str(model_path))
        self.models["stance"] = model
        self.tokenizers["stance"] = tokenizer
    
    def get_model(self, model_type: str) -> Optional[Any]:
        """Get a loaded model by type."""
        return self.models.get(model_type)
    
    def get_tokenizer(self, model_type: str) -> Optional[Any]:
        """Get a tokenizer by model type."""
        return self.tokenizers.get(model_type)
    
    def unload_model(self, model_type: str) -> None:
        """Unload a specific model to free memory."""
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.tokenizers:
                del self.tokenizers[model_type]
            WIBAModelLoader.cleanup_gpu_memory()
            logger.info(f"Unloaded {model_type} model")
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        self.models.clear()
        self.tokenizers.clear()
        WIBAModelLoader.cleanup_gpu_memory()
        logger.info("Unloaded all models")
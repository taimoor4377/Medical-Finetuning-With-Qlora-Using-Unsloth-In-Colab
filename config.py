from dataclasses import dataclass
from transformers import BitsAndBytesConfig
from peft import LoraConfig

@dataclass
class QLoRAConfig:
    model_name: str
    rank: int = 16
    alpha: int = 32

    def configure_quantization(self) -> BitsAndBytesConfig:
        """Return 4-bit quantization settings"""
        return BitsAndBytesConfig(load_in_4bit=True)

    def configure_lora_params(self) -> LoraConfig:
        """Return LoRA adapter configuration"""
        return LoraConfig(r=self.rank, lora_alpha=self.alpha)

    def optimize_for_colab(self) -> None:
        """Adjust parameters for Colab limits"""
        pass

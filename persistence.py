class AdapterSaver:
    def save_adapter(self, model, metadata: dict) -> None:
        """Save fine-tuned adapter with metadata"""
        raise NotImplementedError

    def load_adapter(self, adapter_path: str):
        """Load adapter from file or cloud"""
        raise NotImplementedError

    def verify_compatibility(self, adapter_path: str, base_model: str) -> bool:
        """Check if adapter matches base model"""
        return True

class CheckpointManager:
    def save_checkpoint(self, model, step: int) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str):
        raise NotImplementedError

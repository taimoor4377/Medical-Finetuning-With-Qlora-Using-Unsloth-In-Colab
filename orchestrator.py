from data import MedicalDatasetLoader, MedicalTokenizer
from config import QLoRAConfig
from model import QuantizedModelLoader
from training import QLoRATrainer
from colab_env import ColabResourceDetector
from persistence import AdapterSaver, CheckpointManager
from evaluation import MedicalQueryEvaluator
from errors import ErrorRecoveryManager

class QLoRAWorkflow:
    def __init__(self):
        self.dataset_loader = MedicalDatasetLoader()
        self.config = QLoRAConfig(model_name="llama-3")
        self.model_loader = QuantizedModelLoader()
        self.trainer = None
        self.saver = AdapterSaver()
        self.checkpointer = CheckpointManager()
        self.evaluator = MedicalQueryEvaluator()
        self.error_mgr = ErrorRecoveryManager()
        self.env_detector = ColabResourceDetector()

    def run(self, dataset_path: str):
        """End-to-end fine-tuning pipeline"""
        print("🔍 Loading dataset...")
        dataset = self.dataset_loader.load_dataset(dataset_path)
        self.dataset_loader.validate_format(dataset)
        dataset = self.dataset_loader.preprocess_medical_text(dataset)

        print("⚙️ Configuring QLoRA...")
        qconfig = self.config.configure_quantization()

        print("📦 Loading base model...")
        model = self.model_loader.load_base_model(self.config.model_name, qconfig)

        print("🚀 Starting training...")
        self.trainer = QLoRATrainer(model, None, qconfig)
        try:
            self.trainer.train(dataset)
        except Exception as e:
            print(self.error_mgr.handle_training_errors(e))

        print("💾 Saving adapter...")
        self.saver.save_adapter(model, {"config": vars(self.config)})

        print("✅ Workflow complete.")

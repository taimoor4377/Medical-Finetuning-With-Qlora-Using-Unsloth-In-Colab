class MedicalQueryEvaluator:
    def evaluate_query(self, query: str, model) -> str:
        """Generate response from fine-tuned model"""
        raise NotImplementedError

    def compare_responses(self, query: str, base_model, finetuned_model) -> dict:
        """Compare outputs of base vs fine-tuned models"""
        return {}

    def filter_unsafe_content(self, response: str) -> bool:
        """Detect unsafe or inappropriate content"""
        return False

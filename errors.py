class ErrorRecoveryManager:
    def handle_memory_error(self, error: Exception) -> str:
        return "Reduce batch size or enable gradient checkpointing."

    def handle_session_disconnect(self) -> str:
        return "Reconnect session and reload latest checkpoint."

    def suggest_parameter_adjustments(self, error: Exception) -> list:
        return ["Try smaller rank", "Reduce sequence length"]

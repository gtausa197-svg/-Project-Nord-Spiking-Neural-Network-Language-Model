import torch
import os
from datetime import datetime
from typing import Dict, Any

class SerializationBridge:
    """
    Utility to bundle Isocortex, Allocortex, and Astrocytic states into 
    a single persistent .soul file.
    """
    def __init__(self, kernel):
        self.kernel = kernel

    def save_state(self, filepath: str = None):
        """
        Serializes the Total System Momentum.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"states/kernel_state_{timestamp}.soul"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Collect states from all subsystems
        state_package = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "status": "stable_resume_point"
            },
            "isocortex": self.kernel.isocortex.get_serialized_state(),
            "allocortex": {
                "matrix": self.kernel.allocortex.ca3.memory_matrix.cpu().clone(),
                "usage": self.kernel.allocortex.ca3.usage_counters.cpu().clone()
            },
            "astrocyte": self.kernel.astrocyte.get_metabolic_state()
        }

        torch.save(state_package, filepath)
        print(f"Serialization Bridge: State successfully anchored to {filepath}")

    def resume_state(self, filepath: str):
        """
        Thaws the kernel and restores dynamical momentum.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No state found at {filepath}")

        package = torch.load(filepath)

        # Re-inject states into subsystems
        self.kernel.isocortex.set_serialized_state(package["isocortex"])
        self.kernel.astrocyte.set_metabolic_state(package["astrocyte"])
        
        # Restore Allocortex Memory
        self.kernel.allocortex.ca3.memory_matrix.copy_(package["allocortex"]["matrix"])
        self.kernel.allocortex.ca3.usage_counters.copy_(package["allocortex"]["usage"])

        print(f"Serialization Bridge: Resume complete. Metadata: {package['metadata']}")
      

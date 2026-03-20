import torch
import torch.nn as nn

class LatentWorkspace(nn.Module):
    """
    The equivalent of the 1M-token window for SNN/SSM.
    Maintains the 'Unbroken Thread' as a continuous state vector.
    """
    def __init__(self, d_model: int = 2048, d_state: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # The 'A' matrix: This defines how the past is carried forward.
        # We initialize it to be near-identity to ensure long-term persistence.
        self.A_log = nn.Parameter(torch.log(torch.exp(torch.arange(1, d_state + 1)) - 1))
        
        # The 'Stick-it Note' (Hidden State)
        # This is where the 1M-step equivalent lives.
        self.register_buffer("h_t", torch.zeros(1, d_model, d_state))

    def update_state(self, x_t, delta):
        """
        Updates the internal 'space' with new spikes (x_t).
        delta: The time-step size (Temporal Resolution).
        """
        # Discretize the continuous signal
        A = torch.exp(-torch.exp(self.A_log) * delta)
        
        # Update the hidden state: h_t = A*h_{t-1} + B*x_t
        # This is the 'defragged' carry-forward logic.
        self.h_t = A * self.h_t + (1 - A) * x_t.unsqueeze(-1)
        
        return self.h_t

# --- REPO INTEGRATION ---
# Instead of a 'Token Window' we get a 'State Horizon'.
# As long as the Spectral Radius of A is maintained during Sleep,
# the history remains numerically stable.


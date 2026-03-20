import torch
import torch.nn as nn

class SpikingExecutiveKernel(nn.Module):
    """
    The 'Isocortex' substrate implemented via SpikySpace (arXiv:2601.02411).
    A fully multiplication-free Selective Scan architecture.
    """
    def __init__(self, d_model=2048, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Static dynamics parameter A (quantized to bits for shifting)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        
        # PTSoftplus constants (Lemma 4.1)
        self.x_c = 0.5288
        self.C = 0.9139

    def pt_softplus(self, x):
        """Hardware-friendly Softplus approximation (Equation 11)."""
        return torch.where(x < self.x_c, 2**x, x + self.C)

    def spiking_selective_scan(self, s_t, h_prev, Delta_s, B_t, C_t):
        """
        The core thinking loop (Equation 7 & Algorithm 1).
        Replaces prediction with state-space recurrence.
        """
        # 1. Discrete transition matrix via Bit-Shift (Equation 5)
        # K = floor(A) is used to determine shift magnitude
        K = torch.round(torch.exp(self.A_log))
        
        # If Delta_s (step-size spike) is 1, shift the state; else, preserve it.
        A_bar = torch.where(Delta_s == 1, 2**K, torch.ones_like(K))
        
        # 2. Input projection (Equation 6)
        B_bar = Delta_s * B_t
        
        # 3. State Update: The Integrate-and-Fire Step
        # h_t = SN(A_bar * h_prev + B_bar * s_t)
        # Multiplication-free: A_bar is 2^K (bit-shift), s_t is binary (addition)
        h_t = self.spike_neuron(A_bar * h_prev + B_bar * s_t)
        
        # 4. Output Projection (Equation 8)
        y_t = self.spike_neuron(C_t * h_t)
        
        return y_t, h_t

    def spike_neuron(self, x):
        """Standard IF neuron reset mechanism (Equation 14)."""
        return (x > 1.0).float()
      

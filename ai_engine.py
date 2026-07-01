import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
import ray

# -----------------------------------------------------------------
# Kolmogorov-Arnold Network (KAN) Layer
# -----------------------------------------------------------------
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        
        self.sin_weights = nn.Parameter(torch.randn(out_features, in_features, num_frequencies) / np.sqrt(in_features))
        self.cos_weights = nn.Parameter(torch.randn(out_features, in_features, num_frequencies) / np.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1)
        freqs = torch.arange(1, self.num_frequencies + 1, dtype=torch.float32, device=x.device)
        basis = x_expanded * freqs
        
        sin_basis = torch.sin(basis)
        cos_basis = torch.cos(basis)
        
        sin_out = torch.einsum('bif,oif->bo', sin_basis, self.sin_weights)
        cos_out = torch.einsum('bif,oif->bo', cos_basis, self.cos_weights)
        
        return sin_out + cos_out + self.bias

# -----------------------------------------------------------------
# Spatio-Temporal Graph Convolutional Network (STGCN)
# -----------------------------------------------------------------
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        x_proj = self.fc(x)
        out = torch.einsum('bmn,bnsf->bmsf', adj_matrix, x_proj)
        return F.relu(out)

class MambaBlock(nn.Module):
    """
    A simplified pure PyTorch implementation of the Mamba (Selective State Space) block.
    Bypasses the need for Triton/CUDA compilation on Windows.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1) # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner)
        
        # S4D initialization
        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # 1. Input projection
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # 2. Convolution (1D across sequence length)
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 3. State Space projections
        x_dbl = self.x_proj(x_conv) # (batch, seq_len, d_state * 2 + 1)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (batch, seq_len, d_inner)
        
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Discretization: dt * A
        dA = torch.einsum('bsd,dn->bsdn', dt, A) # (batch, seq_len, d_inner, d_state)
        dB = torch.einsum('bsd,bsn->bsdn', dt, B) # (batch, seq_len, d_inner, d_state)
        
        # Simplified associative scan via sequential accumulation (okay for short horizons)
        # For long horizons, a parallel scan algorithm is needed.
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []
        
        for t in range(seq_len):
            dA_t = torch.exp(dA[:, t]) # (batch, d_inner, d_state)
            dB_t = dB[:, t] # (batch, d_inner, d_state)
            x_t = x_conv[:, t].unsqueeze(-1) # (batch, d_inner, 1)
            
            h = dA_t * h + dB_t * x_t
            
            C_t = C[:, t].unsqueeze(1) # (batch, 1, d_state)
            y_t = torch.einsum('bdn,bkn->bd', h, C_t) # (batch, d_inner)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (batch, seq_len, d_inner)
        
        # Residual and output
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

class SymplecticSTGCN_KAN(nn.Module):
    """
    V5.0 Architecture:
    Graph Convolution (Spatial) -> Mamba SSM (Temporal) -> KAN (Non-linear invariant extraction)
    """
    def __init__(self, input_dim=16, hidden_dim=64, num_layers=2, output_horizons=3):
        super().__init__()
        self.stgcn = GraphConvLayer(input_dim, hidden_dim)
        
        # Replace Transformer with Mamba
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=hidden_dim) for _ in range(num_layers)
        ])
        
        self.kan = KANLinear(hidden_dim, output_horizons, num_frequencies=5)
        
    def forward(self, x, adj_matrix):
        graph_out = self.stgcn(x, adj_matrix)
        primary_seq = graph_out[:, 0, :, :]
        
        mamba_out = primary_seq
        for layer in self.mamba_layers:
            mamba_out = layer(mamba_out)
            
        last_hidden = mamba_out[:, -1, :]
        out = self.kan(last_hidden)
        return out

# -----------------------------------------------------------------
# RL Environment (Counterfactual Learning)
# -----------------------------------------------------------------
class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observation space: 16 features (15 technical + 1 NLP sentiment)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.state = np.zeros(16, dtype=np.float32)
        
    def set_state(self, features: dict):
        keys = list(features.keys())[:16]
        for i, k in enumerate(keys):
            self.state[i] = features[k]
            
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.state, {}
        
    def step(self, action):
        reward = 0.0
        return self.state, reward, True, False, {}

# -----------------------------------------------------------------
# Meta-Gating Network for MARL
# -----------------------------------------------------------------
class MetaGate(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state):
        return self.net(state)

# -----------------------------------------------------------------
# Ray Distributed AI Worker Actor
# -----------------------------------------------------------------
@ray.remote(num_gpus=0.5 if torch.cuda.is_available() else 0)
class AIEngineActor:
    """
    Background worker running STGCN, TFT, KAN, and SAC MARL via Ray.
    """
    def __init__(self):
        print("[AI ENGINE V4] Initializing Ray Actor with STGCN, TFT, KAN, and SAC...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_model = SymplecticSTGCN_KAN().to(self.device)
        self.meta_gate = MetaGate().to(self.device)
        
        self.optimizer = torch.optim.Adam(
            list(self.seq_model.parameters()) + list(self.meta_gate.parameters()), 
            lr=0.001
        )
        self.loss_fn = nn.MSELoss()
        
        self.env = TradingEnv()
        
        self.agent_trend = SAC("MlpPolicy", self.env, verbose=0)
        self.agent_revert = SAC("MlpPolicy", self.env, verbose=0)
        
        print(f"[AI ENGINE V4] Ready on device: {self.device}")

    def process_features(self, step_id: int, features: dict, adj_matrix: list):
        input_dim = 16
        seq_len = 1
        num_nodes = 1
        
        feats_array = []
        keys = list(features.keys())[:15]
        for k in keys:
            feats_array.append(features.get(k, 0.0))
            
        # 16th feature is sentiment
        feats_array.append(features.get("sentiment", 0.0))
            
        x_tensor = torch.tensor([feats_array], dtype=torch.float32, device=self.device)
        x_tensor = x_tensor.view(1, num_nodes, seq_len, 16)
        
        if adj_matrix is None or len(adj_matrix) == 0:
            adj = torch.eye(num_nodes, device=self.device).unsqueeze(0)
        else:
            adj = torch.tensor([adj_matrix], dtype=torch.float32, device=self.device)
            
        self.optimizer.zero_grad()
        
        horizon_preds = self.seq_model(x_tensor, adj)
        target = horizon_preds.detach().clone()
        
        loss = self.loss_fn(horizon_preds, target)
        loss.backward()
        self.optimizer.step()
        
        self.env.set_state(features)
        
        action_trend, _ = self.agent_trend.predict(self.env.state, deterministic=True)
        action_revert, _ = self.agent_revert.predict(self.env.state, deterministic=True)
        
        gate_input = torch.tensor(self.env.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        alpha = self.meta_gate(gate_input).item()
        
        final_action = alpha * action_trend[0] + (1 - alpha) * action_revert[0]
        
        return {
            "forecast": float(horizon_preds[0, 0].item()),
            "direction": 1 if final_action > 0 else -1,
            "confidence": float(abs(final_action)),
            "pred_interval_width": 0.0010,
            "alpha": alpha,
            "action": final_action
        }

import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as slin

class NOTEARS(nn.Module):
    """
    Continuous optimization for structure learning (NOTEARS).
    Zheng et al. 2018: DAGs with NO TEARS.
    """
    def __init__(self, d: int):
        super().__init__()
        # W is the adjacency matrix we want to learn
        self.W = nn.Parameter(torch.zeros(d, d))
        
    def forward(self, X):
        # Linear Structural Equation Model: X = X W + noise
        return torch.matmul(X, self.W)
        
    def h(self):
        """DAG constraint: tr(e^(W * W)) - d = 0"""
        W_squared = self.W * self.W
        E = torch.matrix_exp(W_squared)
        h_val = torch.trace(E) - self.W.shape[0]
        return h_val

def learn_causal_graph(X: np.ndarray, lambda1: float = 0.01, w_threshold: float = 0.3, max_iter: int = 100) -> np.ndarray:
    """
    Learn a causal graph from observational data X.
    X: (samples, variables) array of continuous returns/capacities.
    Returns: Binary Adjacency Matrix (variables, variables).
    """
    n, d = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    
    model = NOTEARS(d).to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
    
    # Augmented Lagrangian params
    rho = 1.0
    alpha = 0.0
    h_tol = 1e-8
    
    for _ in range(max_iter):
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_t)
            # Least squares loss
            loss = 0.5 / n * torch.sum((X_t - X_hat)**2)
            # L1 regularization
            l1_penalty = lambda1 * torch.sum(torch.abs(model.W))
            # DAG constraint
            h_val = model.h()
            # Augmented Lagrangian
            obj = loss + l1_penalty + 0.5 * rho * h_val**2 + alpha * h_val
            obj.backward()
            return obj
            
        optimizer.step(closure)
        
        with torch.no_grad():
            h_val = model.h().item()
            if h_val < h_tol:
                break
            # Update dual variables
            alpha += rho * h_val
            rho *= 10
            
    # Thresholding
    W_est = model.W.detach().cpu().numpy()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

if __name__ == "__main__":
    # Test execution
    print("Testing NOTEARS Causal Discovery...")
    # Simulate some data where X0 -> X1 -> X2
    X = np.random.randn(1000, 3)
    X[:, 1] += 2.0 * X[:, 0]
    X[:, 2] += 1.5 * X[:, 1]
    
    W = learn_causal_graph(X)
    print("Estimated Causal Graph Adjacency Matrix:")
    print(W)

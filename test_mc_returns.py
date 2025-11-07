#!/usr/bin/env python3
"""
Quick test to verify Monte Carlo returns are computed correctly.
"""
import torch
import numpy as np

def compute_mc_returns(R, M, gamma=1.0):
    """Compute Monte Carlo returns the same way as in _update_baseline"""
    device = R.device
    T, B = R.shape
    G = torch.zeros_like(R)
    G_running = torch.zeros(B, device=device)

    # Backward pass to compute returns
    for t in range(T - 1, -1, -1):
        G_running = R[t] * M[t] + gamma * G_running
        G[t] = G_running

    return G

# Test case 1: Simple rewards at the end
print("Test 1: Reward only at final timestep")
R = torch.tensor([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, -1.0]
])
M = torch.ones_like(R)
G = compute_mc_returns(R, M, gamma=1.0)
print(f"Rewards:\n{R}")
print(f"Returns:\n{G}")
print(f"Expected: All timesteps should have return = final reward")
assert torch.allclose(G[:, 0], torch.tensor([1.0, 1.0, 1.0]))
assert torch.allclose(G[:, 1], torch.tensor([-1.0, -1.0, -1.0]))
print("✓ PASSED\n")

# Test case 2: Rewards at decision point
print("Test 2: Reward at decision timestep (like the task)")
R = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, -1.0],
    [0.0, 0.0]
])
M = torch.ones_like(R)
G = compute_mc_returns(R, M, gamma=1.0)
print(f"Rewards:\n{R}")
print(f"Returns:\n{G}")
print("Expected:")
print("  Trial 0: G[0]=1, G[1]=1, G[2]=0, G[3]=0, G[4]=0")
print("  Trial 1: G[0]=-1, G[1]=-1, G[2]=-1, G[3]=-1, G[4]=0")
assert torch.allclose(G[:, 0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]))
assert torch.allclose(G[:, 1], torch.tensor([-1.0, -1.0, -1.0, -1.0, 0.0]))
print("✓ PASSED\n")

# Test case 3: With discounting
print("Test 3: With gamma=0.9")
R = torch.tensor([
    [0.0],
    [0.0],
    [1.0]
])
M = torch.ones_like(R)
gamma = 0.9
G = compute_mc_returns(R, M, gamma=gamma)
print(f"Rewards:\n{R}")
print(f"Returns:\n{G}")
expected_g0 = gamma**2 * 1.0  # discounted by 2 steps
expected_g1 = gamma * 1.0      # discounted by 1 step
expected_g2 = 1.0              # no discounting
print(f"Expected: G[0]={expected_g0:.3f}, G[1]={expected_g1:.3f}, G[2]={expected_g2:.3f}")
assert torch.allclose(G, torch.tensor([[expected_g0], [expected_g1], [expected_g2]]))
print("✓ PASSED\n")

# Test case 4: With masking (episode ends early)
print("Test 4: With masking (episode ends at t=2)")
R = torch.tensor([
    [0.0],
    [1.0],
    [0.0],
    [0.0],
    [0.0]
])
M = torch.tensor([
    [1.0],
    [1.0],
    [0.0],  # Episode ended after t=1
    [0.0],
    [0.0]
])
G = compute_mc_returns(R, M, gamma=1.0)
print(f"Rewards:\n{R}")
print(f"Mask:\n{M}")
print(f"Returns:\n{G}")
print("Expected: Only masked rewards contribute")
print("  G[0] = R[0]*M[0] + R[1]*M[1] + R[2]*M[2] + ... = 0*1 + 1*1 + 0*0 = 1.0")
print("  G[1] = R[1]*M[1] + R[2]*M[2] + ... = 1*1 + 0*0 = 1.0")
print("  G[2] = R[2]*M[2] + ... = 0*0 = 0.0")
assert torch.allclose(G, torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0]]))
print("✓ PASSED\n")

# Test case 5: Reward at invalid timestep should not contribute
print("Test 5: Reward at masked timestep should NOT contribute")
R = torch.tensor([
    [0.0],
    [1.0],
    [5.0],  # This reward is at invalid timestep!
    [0.0]
])
M = torch.tensor([
    [1.0],
    [1.0],
    [0.0],  # Episode ended, this timestep is invalid
    [0.0]
])
G = compute_mc_returns(R, M, gamma=1.0)
print(f"Rewards:\n{R}")
print(f"Mask:\n{M}")
print(f"Returns:\n{G}")
print("Expected: R[2]=5.0 should NOT contribute because M[2]=0")
print("  G[0] = 0*1 + 1*1 + 5*0 + 0*0 = 1.0")
print("  G[1] = 1*1 + 5*0 + 0*0 = 1.0")
assert torch.allclose(G, torch.tensor([[1.0], [1.0], [0.0], [0.0]]))
print("✓ PASSED\n")

print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nThe Monte Carlo returns are being computed correctly.")
print("This means the value network will now learn from ACTUAL future rewards")
print("instead of trying to bootstrap from its own (initially random) predictions!")

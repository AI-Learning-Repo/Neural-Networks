import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("PyTorch version:", torch.__version__)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

# XOR data
X_np = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1]
], dtype=np.float32)
y_np = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Convert NumPy arrays to PyTorch Tensors
X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Set random seed for reproducibility
seed = 32
torch.manual_seed(seed)
np.random.seed(seed)

# Sequential Model (2 -> 4 -> 1)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

print("\nModel Architecture:")
print(model)

# Count parameters (12 + 5 = 17)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}\n")

# Loss and Optimizer (Keras Adam default learning rate is 0.001)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (1000 epochs, full batch / batch_size=4)
model.train()
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Evaluate training performance
model.eval()
with torch.no_grad():
    scores = model(X)
    train_loss = criterion(scores, y).item()
    preds = (scores > 0.5).float()
    train_accuracy = (preds == y).float().mean().item()

print(f"Final training loss: {train_loss:.4f}")
print(f"Final training accuracy: {train_accuracy:.0%}")

# Inspect every example
scores_np = scores.squeeze().numpy()
predictions = (scores_np > 0.5).astype(int)

print("\nXOR predictions:")
for inputs, score, predicted, expected in zip(
    X_np, scores_np, predictions, y_np.reshape(-1)
):
    print(
        f"Input: {inputs.astype(int)} | "
        f"Score: {score:.3f} | "
        f"Predicted: {predicted} | Expected: {int(expected)}"
    )

calculated_accuracy = np.mean(predictions == y_np.reshape(-1))
print(f"Accuracy calculated from predictions: {calculated_accuracy:.0%}")
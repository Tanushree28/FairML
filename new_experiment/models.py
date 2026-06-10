"""Models and training loop for the new experiment.

Differences from the original Models.py:
- `forward` returns logits (no sigmoid) — the loss applies it once, fixing the
  double-sigmoid bug.
- The `num_epochs` argument is actually used (the original shadowed it).
- Early stopping on validation loss, with a deep-copied best-weights snapshot
  (the original EarlyStopping.py stored a live reference to state_dict, which
  keeps updating as training continues).
- Adds a small MLP as a second architecture next to logistic regression.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim

from losses import composite_loss


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(model, X_train, y_train, group_train, X_val, y_val, group_val,
                alpha, beta, num_epochs=300, lr=0.01, weight_decay=1e-4,
                patience=20, verbose=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = composite_loss(model(X_train), y_train, group_train, alpha, beta)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = composite_loss(model(X_val), y_val, group_val, alpha, beta).item()
        stopper(val_loss, model)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch [{epoch + 1}/{num_epochs}], train loss: {loss.item():.4f}, val loss: {val_loss:.4f}")
        if stopper.early_stop:
            break

    stopper.load_best_model(model)
    return model


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X)).numpy().ravel()

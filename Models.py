import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def train_model(self, criterion, X_train_tensorized, y_train_tensorized, sensitive_features = None, group_fairness = None, individual_fairness = None, alpha = None, beta = None, epoch = 100):
        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4) 

        num_epochs = 100
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(X_train_tensorized)

            if sensitive_features is not None and alpha is not None and beta is not None:
                loss = criterion(y_train_tensorized, outputs, sensitive_features, group_fairness, individual_fairness, alpha, beta)
            else:
                loss = criterion(outputs, y_train_tensorized)
            
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X, y):
        self.eval()

        with torch.no_grad():
            y_pred = self(X).round()
            accuracy = (y_pred.numpy() == y.numpy()).mean()
            print(f'LogReg Model Accuracy: {accuracy:.4f}')

    def print_metric_frame(self, y_true, y_pred, sensitive_features):
        metric_frame = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,
                'True Positive Rate': true_positive_rate,
                'False Positive Rate': false_positive_rate
            },

            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )

        print(metric_frame.by_group)
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

# ipca.py

import torch

class IncrementalPCA:
    """
    Implements an incremental PCA using a method similar to CCIPCA for online learning.
    This allows updating the PCA model one batch at a time without storing all data.
    """
    def __init__(self, n_components, device='cpu'):
        self.n_components = n_components
        self.device = device
        self.n_samples_seen = 0
        self.mean = None
        self.components = None # Principal axes in feature space
        self.explained_variance = None # Variance of data projected onto each component

    def fit(self, X):
        """Initializes PCA with the first batch of data."""
        if X.shape[0] < self.n_components:
            raise ValueError("First batch must have at least as many samples as n_components.")
        
        self.n_samples_seen = X.shape[0]
        self.mean = torch.mean(X, dim=0)
        X_centered = X - self.mean
        
        # Use SVD for the initial batch
        U, S, V = torch.svd(X_centered)
        self.components = V[:, :self.n_components].T # (n_components, n_features)
        
        # Variance is (S^2 / (n-1))
        self.explained_variance = (S[:self.n_components] ** 2) / (self.n_samples_seen - 1)

    def update(self, X):
        """Incrementally updates the PCA model with a new batch of data."""
        if self.n_samples_seen == 0:
            self.fit(X)
            return

        n_new_samples = X.shape[0]
        n_total_samples = self.n_samples_seen + n_new_samples
        
        # Update mean
        old_mean = self.mean.clone()
        self.mean = (self.n_samples_seen * old_mean + torch.sum(X, dim=0)) / n_total_samples
        
        # Center new data with the updated mean
        X_centered = X - self.mean
        
        # Project centered new data onto existing components
        projections = torch.matmul(X_centered, self.components.T)
        
        # Reconstruct and find residuals (part of new data not explained by old components)
        reconstructed = torch.matmul(projections, self.components)
        residuals = X_centered - reconstructed
        
        # Update components with the residuals via QR decomposition
        # This is a key step in many incremental PCA algorithms
        Q, R = torch.linalg.qr(residuals.T)
        
        # Combine old and new information
        combined_matrix = torch.block_diag(
            torch.diag(self.explained_variance * (self.n_samples_seen - 1)),
            torch.matmul(projections.T, projections)
        )
        T = torch.cat((torch.zeros(self.n_components, Q.shape[1], device=self.device), R), dim=0)
        
        # Find the new principal components via SVD on a smaller combined matrix
        U_hat, S_hat, V_hat = torch.svd(torch.cat((combined_matrix, T), dim=1))
        
        self.components = torch.matmul(V_hat[:, :self.n_components].T, torch.cat((self.components, Q.T), dim=0))
        self.explained_variance = (S_hat[:self.n_components] ** 2) / (n_total_samples - 1)
        self.n_samples_seen = n_total_samples

    def sample(self, n_samples):
        """Generates new samples from the learned distribution."""
        if self.components is None:
            raise RuntimeError("PCA has not been fitted yet.")
        
        # Sample random coefficients from a Gaussian distribution scaled by the explained variance
        coeffs = torch.randn(n_samples, self.n_components, device=self.device) * torch.sqrt(self.explained_variance)
        
        # Generate samples by combining the mean with the scaled principal components
        synthetic_features = self.mean + torch.matmul(coeffs, self.components)
        return synthetic_features

    def state_dict(self):
        return {
            'n_components': self.n_components,
            'n_samples_seen': self.n_samples_seen,
            'mean': self.mean,
            'components': self.components,
            'explained_variance': self.explained_variance
        }

    def load_state_dict(self, state_dict):
        self.n_components = state_dict['n_components']
        self.n_samples_seen = state_dict['n_samples_seen']
        self.mean = state_dict['mean']
        self.components = state_dict['components']
        self.explained_variance = state_dict['explained_variance']
        self.device = self.mean.device
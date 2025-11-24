"""
Generative Classification with Beta KDE
=======================================

This example demonstrates how to build a flexible, non-parametric Generative Classifier
using BetaKDE. By modeling the probability density P(x|y) for each class using
BetaKDE, we can use Bayes' rule to predict the class posterior P(y|x).

This approach has several advantages:
1. It naturally handles bounded data (e.g., [0, 1] or physical limits).
2. It can model complex, non-linear decision boundaries.
3. It supports multivariate data via the Beta Copula.
"""

import sys
import os

# Add the ../src directory to Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.insert(0, src_path)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from beta_kde import BetaKDE

class BetaKDEClassifier(BaseEstimator, ClassifierMixin):
    """
    A Generative Classifier that uses BetaKDE to estimate class-conditional densities.
    
    Parameters
    ----------
    bandwidth : str or float, default='beta-reference'
        The bandwidth selection method or fixed value for the BetaKDE.
    bounds : tuple, default=(0, 1)
        The strict bounds of the support (min, max).
    """
    def __init__(self, bandwidth='beta-reference', bounds=(0, 1)):
        self.bandwidth = bandwidth
        self.bounds = bounds

    def fit(self, X, y):
        """
        Fit a separate BetaKDE model for each unique class in y.
        """
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        
        for cls in self.classes_:
            X_cls = X[y == cls]
            
            # Fit a BetaKDE for this specific class
            # We pass the bounds explicitly to handle arbitrary support ranges
            kde = BetaKDE(bandwidth=self.bandwidth, bounds=self.bounds)
            kde.fit(X_cls)
            
            self.models_[cls] = kde
            self.priors_[cls] = len(X_cls) / len(X)
            
        return self

    def predict_log_proba(self, X):
        """
        Predict log posterior probabilities for each class.
        log P(y|x) ∝ log P(x|y) + log P(y)
        """
        log_probs = []
        for cls in self.classes_:
            # CRITICAL: normalized=True is required here!
            # We are comparing probabilities between different models (classes).
            # If we used raw density values (normalized=False), a model with a 
            # smaller bandwidth would artificially dominate because its peak 
            # would be "taller" (larger area under the curve).
            log_likelihood = self.models_[cls].score_samples(X, normalized=True)
            log_prior = np.log(self.priors_[cls])
            
            log_posterior = log_likelihood + log_prior
            log_probs.append(log_posterior)
        
        return np.array(log_probs).T

    def predict(self, X):
        """Predict the class label with the highest posterior probability."""
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


if __name__ == "__main__":
    # -----------------------------------------------------------
    # Demo: Classifying the "Moons" dataset (Non-linear & Bounded)
    # -----------------------------------------------------------
    print("Generating data...")
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    # Define strict bounds for the estimator. 
    # BetaKDE needs to know the absolute limits of the data support.
    # Here, we estimate them from the data with a small padding.
    bounds = (X.min() - 0.1, X.max() + 0.1)
    print(f"Data Bounds determined as: {bounds}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Initialize and Train
    print("Training BetaKDEClassifier...")
    clf = BetaKDEClassifier(bandwidth='beta-reference', bounds=bounds)
    clf.fit(X_train, y_train)

    # Evaluate
    acc = clf.score(X_test, y_test)
    print(f"Test Set Accuracy: {acc:.2%}")

    # Plotting
    print("Plotting decision boundary...")
    try:
        # Create a meshgrid to visualize decision regions
        xx, yy = np.meshgrid(
            np.linspace(bounds[0], bounds[1], 100),
            np.linspace(bounds[0], bounds[1], 100)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict for the whole grid
        Z = clf.predict(grid_points).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolor='k', alpha=0.7)
        plt.title(f"Beta KDE Generative Classifier (Accuracy: {acc:.2%})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")
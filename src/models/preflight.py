import numpy as np

def adjust_prior(p_sampled, p_true_prior, p_train_prior):
    # Avoid division by zero
    p_sampled = np.clip(p_sampled, 1e-15, 1 - 1e-15)
    
    odds_sampled = p_sampled / (1.0 - p_sampled)
    odds_true_prior = p_true_prior / (1.0 - p_true_prior)
    odds_train_prior = p_train_prior / (1.0 - p_train_prior)
    odds_adjusted = odds_sampled * (odds_true_prior / odds_train_prior)
    return odds_adjusted / (1.0 + odds_adjusted)

class PriorShiftedCalibratedModel:
    def __init__(self, calibrated_model, true_prior, train_prior, features):
        self.calibrated_model = calibrated_model
        self.true_prior = true_prior
        self.train_prior = train_prior
        self.features = features
        
    def predict_proba(self, X):
        probs = self.calibrated_model.predict_proba(X)
        p_sampled = probs[:, 1]
        p_adjusted = adjust_prior(p_sampled, self.true_prior, self.train_prior)
        return np.vstack([1 - p_adjusted, p_adjusted]).T
        
    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)[:, 1]
        if threshold is None:
            threshold = self.true_prior * 5
        return (probs >= threshold).astype(int)

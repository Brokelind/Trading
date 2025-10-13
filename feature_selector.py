import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

class FeatureSelector:
    """
    Feature selection utilities for trading models
    """
    
    def __init__(self, method='mutual_info', n_features=50):
        """
        Args:
            method: 'mutual_info', 'f_test', 'rfe', 'importance', 'correlation', 'combined'
            n_features: Number of features to select
        """
        self.method = method
        self.n_features = n_features
        self.selected_features_ = None
        self.feature_scores_ = None
        self.selector_ = None
        
    def fit(self, X, y, feature_names=None):
        """
        Fit feature selector
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable
            feature_names: List of feature names
        """
        import pandas as pd  # Import here to be safe
        import numpy as np
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        self.feature_names_ = feature_names
        
        if self.method == 'mutual_info':
            self._mutual_info_selection(X, y)
        elif self.method == 'f_test':
            self._f_test_selection(X, y)
        elif self.method == 'rfe':
            self._rfe_selection(X, y)
        elif self.method == 'importance':
            self._importance_selection(X, y)
        elif self.method == 'correlation':
            self._correlation_selection(X, y)
        elif self.method == 'combined':
            self._combined_selection(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _mutual_info_selection(self, X, y):
        """Mutual information feature selection"""
        import pandas as pd
        from sklearn.feature_selection import SelectKBest, mutual_info_regression
        
        selector = SelectKBest(
            score_func=mutual_info_regression,
            k=min(self.n_features, X.shape[1])
        )
        selector.fit(X, y.ravel())
        
        self.selector_ = selector
        self.feature_scores_ = pd.DataFrame({
            'feature': self.feature_names_,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features_ = self.feature_scores_.head(self.n_features)['feature'].tolist()
    
    def _f_test_selection(self, X, y):
        """F-test feature selection"""
        import pandas as pd
        from sklearn.feature_selection import SelectKBest, f_regression
        
        selector = SelectKBest(
            score_func=f_regression,
            k=min(self.n_features, X.shape[1])
        )
        selector.fit(X, y.ravel())
        
        self.selector_ = selector
        self.feature_scores_ = pd.DataFrame({
            'feature': self.feature_names_,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features_ = self.feature_scores_.head(self.n_features)['feature'].tolist()
    
    def _rfe_selection(self, X, y):
        """Recursive Feature Elimination"""
        import pandas as pd
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(self.n_features, X.shape[1]),
            step=5
        )
        selector.fit(X, y.ravel())
        
        self.selector_ = selector
        self.feature_scores_ = pd.DataFrame({
            'feature': self.feature_names_,
            'rank': selector.ranking_
        }).sort_values('rank')
        
        self.selected_features_ = self.feature_scores_.head(self.n_features)['feature'].tolist()
    
    def _importance_selection(self, X, y):
        """Feature importance from Random Forest"""
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y.ravel())
        
        self.feature_scores_ = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.selected_features_ = self.feature_scores_.head(self.n_features)['feature'].tolist()
    
    def _correlation_selection(self, X, y):
        """Correlation-based selection with multicollinearity removal"""
        import pandas as pd
        
        df = pd.DataFrame(X, columns=self.feature_names_)
        df['target'] = y
        
        # Calculate correlation with target
        correlations = df.corr()['target'].drop('target').abs()
        
        # Sort by correlation
        sorted_features = correlations.sort_values(ascending=False)
        
        # Remove highly correlated features
        selected = []
        for feat in sorted_features.index:
            if len(selected) >= self.n_features:
                break
            
            # Check correlation with already selected features
            if len(selected) == 0:
                selected.append(feat)
            else:
                max_corr = df[selected + [feat]].corr()[feat].drop(feat).abs().max()
                if max_corr < 0.85:  # Threshold for multicollinearity
                    selected.append(feat)
        
        self.selected_features_ = selected
        self.feature_scores_ = pd.DataFrame({
            'feature': sorted_features.index,
            'correlation': sorted_features.values
        })
    
    def _combined_selection(self, X, y):
        """Ensemble of multiple selection methods"""
        methods = ['mutual_info', 'f_test', 'importance']
        all_selections = []
        
        for method in methods:
            try:
                temp_selector = FeatureSelector(method=method, n_features=self.n_features * 2)
                temp_selector.fit(X, y, self.feature_names_)
                all_selections.append(set(temp_selector.selected_features_))
            except Exception as e:
                logger.warning(f"Method {method} failed in combined selection: {e}")
                continue
        
        if not all_selections:
            # Fallback to importance if all methods fail
            logger.warning("All methods failed, falling back to importance only")
            self._importance_selection(X, y)
            return
        
        # Vote: features that appear in at least 2 methods (or 1 if only 1 method succeeded)
        feature_votes = {}
        for features in all_selections:
            for feat in features:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1
        
        # Sort by votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f for f, v in sorted_features[:self.n_features]]
        
        import pandas as pd
        self.feature_scores_ = pd.DataFrame(sorted_features, columns=['feature', 'votes'])
    
    def transform(self, X, feature_names=None):
        """Transform data to selected features"""
        import pandas as pd
        
        if feature_names is None:
            feature_names = self.feature_names_
        
        df = pd.DataFrame(X, columns=feature_names)
        return df[self.selected_features_].values
    
    def fit_transform(self, X, y, feature_names=None):
        """Fit and transform"""
        self.fit(X, y, feature_names)
        return self.transform(X, feature_names)
    
    def plot_feature_scores(self, top_n=30, figsize=(12, 8)):
        """Visualize feature scores"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.feature_scores_ is None:
            print("No feature scores available. Run fit() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Get top N features
        top_features = self.feature_scores_.head(top_n)
        
        # Create bar plot
        score_col = [c for c in top_features.columns if c != 'feature'][0]
        
        sns.barplot(data=top_features, y='feature', x=score_col, palette='viridis')
        plt.title(f'Top {top_n} Features ({self.method})')
        plt.xlabel('Score')
        plt.ylabel('Feature')
        plt.tight_layout()
"""
Core Preprocessing Engine using Strategy, Observer Patterns, and Dynamic Programming.
"""
import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Any
from abc import ABC, abstractmethod

# ==========================================
# DESIGN PATTERN: Strategy Pattern for Scalers
# ==========================================
class ScalingStrategy(ABC):
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        pass

class StandardScalerStrategy(ScalingStrategy):
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        res = df.copy()
        for col in columns:
            mean = res[col].mean()
            std = res[col].std()
            res[col] = (res[col] - mean) / (std if std != 0 else 1)
        return res

class MinMaxScalerStrategy(ScalingStrategy):
    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        res = df.copy()
        for col in columns:
            min_val = res[col].min()
            max_val = res[col].max()
            res[col] = (res[col] - min_val) / ((max_val - min_val) if (max_val - min_val) != 0 else 1)
        return res

class ImputerStrategy:
    """Strategy class for filling missing values."""
    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        res = df.copy()
        for col in columns:
            if self.strategy == "mean":
                res[col] = res[col].fillna(res[col].mean())
            elif self.strategy == "median":
                res[col] = res[col].fillna(res[col].median())
        return res

# ==========================================
# DESIGN PATTERN: Observer Pattern for Pipeline Monitoring
# ==========================================
class PipelineObserver(ABC):
    @abstractmethod
    def update(self, step_name: str, status: str) -> None:
        pass

class ConsoleLogger(PipelineObserver):
    def update(self, step_name: str, status: str) -> None:
        print(f"[PIPELINE NOTIFICATION] Step '{step_name}' completed with status: {status}")

# ==========================================
# SOLID & Architectural Pattern: Pipeline System
# ==========================================
class Pipeline:
    """Architectural Pattern: Pipeline Architecture. Manages sequential operations."""
    def __init__(self):
        self._steps: List[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = []
        self._observers: List[PipelineObserver] = []

    def attach_observer(self, observer: PipelineObserver) -> None:
        self._observers.append(observer)

    def _notify(self, step_name: str, status: str) -> None:
        for observer in self._observers:
            observer.update(step_name, status)

    def add_step(self, name: str, transformer: Callable[[pd.DataFrame], pd.DataFrame]) -> 'Pipeline':
        self._steps.append((name, transformer))
        return self

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        current_df = df.copy()
        for name, step in self._steps:
            current_df = step(current_df)
            self._notify(name, "SUCCESS")
        return current_df
    


# ==========================================
# ADVANCED CONCEPT: Dynamic Programming (DP)
# ==========================================
class DPOptimalFeatureSelector:
    """
    Uses Dynamic Programming (0/1 Knapsack logic) to select the optimal subset of 
    features based on their importance weights vs computational cost bounds.
    """
    @staticmethod
    def select_features(features: List[str], importances: List[int], costs: List[int], max_cost: int) -> List[str]:
        n = len(features)
        # DP table initialization
        dp = [[0 for _ in range(max_cost + 1)] for _ in range(n + 1)]
        
        # Bottom-up DP table building
        for i in range(1, n + 1):
            for w in range(1, max_cost + 1):
                if costs[i-1] <= w:
                    dp[i][w] = max(importances[i-1] + dp[i-1][w-costs[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Backtracking to find selected features
        res = dp[n][max_cost]
        w = max_cost
        selected_features = []
        
        for i in range(n, 0, -1):
            if res <= 0:
                break
            if res == dp[i-1][w]:
                continue
            else:
                selected_features.append(features[i-1])
                res -= importances[i-1]
                w -= costs[i-1]
                
        return selected_features
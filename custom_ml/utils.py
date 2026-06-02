"""
Utility Layer demonstrating Functional Programming and Recursion.
"""
import pandas as pd
from typing import List, Callable, Any

class FunctionalTransformer:
    """
    ADVANCED CONCEPT: Functional Programming.
    Uses pure functions, map/reduce philosophies, and composable pipelines.
    """
    @staticmethod
    def compose(*functions: Callable[[pd.DataFrame], pd.DataFrame]) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Composes multiple functions into a single functional pipeline wrapper."""
        def inner(df: pd.DataFrame) -> pd.DataFrame:
            res = df.copy()
            for f in functions:
                res = f(res)
            return res
        return inner

    @staticmethod
    def apply_math_mapping(df: pd.DataFrame, col: str, func: Callable[[Any], Any]) -> pd.DataFrame:
        """Pure Function that applies a functional mapping via pandas map."""
        res = df.copy()
        res[col] = res[col].map(func)
        return res

# ==========================================
# ADVANCED CONCEPT: Recursion
# ==========================================
class RecursiveTreePruner:
    """
    Simulates operational hierarchy tree traversal (like Decision Trees) 
    recursively to prune low-variance feature nodes.
    """
    @staticmethod
    def prune_hierarchy_tree(node: dict, min_threshold: float) -> dict:
        """Recursively traverses and prunes a dict-based nested tree structure."""
        if "variance" in node and node["variance"] < min_threshold:
            return {}  # Pruned Node
        
        if "children" in node:
            pruned_children = []
            for child in node["children"]:
                pruned_child = RecursiveTreePruner.prune_hierarchy_tree(child, min_threshold)
                if pruned_child:  # If not fully dropped
                    pruned_children.append(pruned_child)
            node["children"] = pruned_children
            
        return node
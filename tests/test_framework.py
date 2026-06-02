import pytest
import pandas as pd
import numpy as np
from custom_ml.core import Pipeline, StandardScalerStrategy, ImputerStrategy, DPOptimalFeatureSelector
from custom_ml.utils import FunctionalTransformer, RecursiveTreePruner

def test_pipeline_and_strategies():
    # Setup dummy data
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan, 4.0],
        'B': [10.0, 20.0, 30.0, 40.0]
    })
    
    pipeline = Pipeline()
    imputer = ImputerStrategy(strategy="mean")
    scaler = StandardScalerStrategy()
    
    pipeline.add_step("Impute A", lambda d: imputer.transform(d, ['A']))
    pipeline.add_step("Scale B", lambda d: scaler.fit_transform(d, ['B']))
    
    processed_df = pipeline.execute(df)
    
    assert processed_df['A'].isna().sum() == 0
    assert np.isclose(processed_df['B'].mean(), 0.0, atol=1e-7)

def test_dynamic_programming():
    features = ["f1", "f2", "f3"]
    importances = [60, 100, 120]
    costs = [10, 20, 30]
    max_cost = 50
    
    selected = DPOptimalFeatureSelector.select_features(features, importances, costs, max_cost)
    assert "f3" in selected
    assert "f2" in selected

def test_recursion():
    tree = {
        "name": "root", "variance": 0.9,
        "children": [
            {"name": "child_1", "variance": 0.1},  # Should be pruned
            {"name": "child_2", "variance": 0.7, "children": []}
        ]
    }
    pruned = RecursiveTreePruner.prune_hierarchy_tree(tree, min_threshold=0.5)
    assert len(pruned["children"]) == 1
    assert pruned["children"][0]["name"] == "child_2"
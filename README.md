# Custom ML Preprocessing & Feature Engineering Framework

This library is a highly modular, clean, and extensible Python machine learning preprocessing library developed for the Advanced Programming course project. 

It implements end-to-end data pipelines incorporating advanced programming paradigms, clean architecture, and standard software patterns.

## 🚀 Installation & Local Setup

To install this package locally in editable/development mode, navigate to the project root directory where `pyproject.toml` is located and run:

```bash
pip install -e .

1. Object-Oriented Programming (OOP)

Where: Located across custom_ml/data.py and custom_ml/core.py
How: Leverages structural Inheritance via BaseLoader(ABC) to enforce interfaces for concrete classes CSVLoader and JSONLoader. High usage of Encapsulation protects private states inside the Pipeline list tracking objects (self._steps)

2. Functional Programming (FP)

Where: Located inside custom_ml/utils.py -> FunctionalTransformer
How: Implements Pure Functions like apply_math_mapping that avoid side effects by explicitly duplicating frames (df.copy()). Includes a functional mathematical combinator pipeline workflow via compose(*functions).

3. Concurrency

Where: Located inside custom_ml/data.py -> ConcurrentDataLoader.
How: Employs multi-threaded workers via concurrent.futures.ThreadPoolExecutor to speed up system operational latency. It handles multi-source I/O operations asynchronously by spinning up non-blocking threads to read discrete source inputs simultaneously.

4. Recursion / Dynamic Programming

Where: * Recursion: custom_ml/utils.py -> RecursiveTreePruner.prune_hierarchy_tree
Dynamic Programming: custom_ml/core.py -> DPOptimalFeatureSelector.select_features
How: Recursion is leveraged to systematically traverse an multi-nested operational variance hierarchy (tree dictionary structure) to prune low variance features recursively. Dynamic Programming applies a classic bottom-up 0/1 Knapsack tabular state transition layout to select optimal feature configurations bounded by strict execution computational budgets.

5. SOLID Principles

Where: Throughout the entire module design boundaries.
How: * Single Responsibility Principle (SRP): data.py handles parsing/loading only, core.py encapsulates engineering workflows, utils.py houses functional tools.
Open/Closed Principle (OCP) & Dependency Inversion Principle (DIP): Algorithms like ScalingStrategy depend heavily on loose abstractions rather than rigid concrete implementations, allowing developers to inject custom math scaling structures without touching core source frameworks.

6. Architectural & Design Patterns

Architectural Pattern: Pipeline Architecture in core.py manages execution sequences fluidly.
Design Patterns Used:
   1- Factory Pattern: Implemented in DataLoaderFactory to generate decoupled system loaders on demand dynamically.
   2- Strategy Pattern: Utilized across ScalingStrategy to seamlessly hot-swap core algebraic data scalers at execution runtime.
   3- Observer Pattern: Configured via PipelineObserver and ConsoleLogger to notify subscriber clients dynamically upon operational task accomplishments.
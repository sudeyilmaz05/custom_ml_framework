"""
Data Loading Layer using Factory Pattern and Concurrency.
"""
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseLoader(ABC):
    """Abstract Base Class for Data Loaders (OOP: Inheritance & Interface)"""
    @abstractmethod
    def load_data(self, source: str) -> pd.DataFrame:
        pass

class CSVLoader(BaseLoader):
    """Concrete Loader for CSV files."""
    def load_data(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source)

class JSONLoader(BaseLoader):
    """Concrete Loader for JSON files."""
    def load_data(self, source: str) -> pd.DataFrame:
        return pd.read_json(source)

class DataLoaderFactory:
    """
    DESIGN PATTERN: Factory Pattern.
    Decouples client from specific file parsing logic.
    """
    @staticmethod
    def get_loader(file_type: str) -> BaseLoader:
        file_type = file_type.lower()
        if file_type == "csv":
            return CSVLoader()
        elif file_type == "json":
            return JSONLoader()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

class ConcurrentDataLoader:
    """
    ADVANCED CONCEPT: Concurrency.
    Loads multiple data files in parallel using ThreadPoolExecutor.
    """
    def __init__(self, sources: Dict[str, str]):
        self.sources = sources  # Format: {"file1": "csv", "file2": "json"}

    def load_all(self) -> List[pd.DataFrame]:
        factory = DataLoaderFactory()
        results: List[pd.DataFrame] = []
        
        # Concurrency implemented here safely
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_source = {
                executor.submit(factory.get_loader(ftype).load_data, path): path 
                for path, ftype in self.sources.items()
            }
            for future in concurrent.futures.as_completed(future_to_source):
                try:
                    data = future.result()
                    results.append(data)
                except Exception as e:
                    print(f"Error loading {future_to_source[future]}: {e}")
        return results
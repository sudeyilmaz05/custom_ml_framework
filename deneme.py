import pandas as pd
import numpy as np
from custom_ml.core import Pipeline, ImputerStrategy, StandardScalerStrategy, ConsoleLogger
from custom_ml.utils import FunctionalTransformer

# 1. Create a dummy data frame with anomalies
data = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50],
    'salary': [50000, 60000, 80000, np.nan, 120000]
})

# 2. Instantiate Components
pipeline = Pipeline()
logger = ConsoleLogger()
pipeline.attach_observer(logger) # Attach observer pattern

imputer = ImputerStrategy(strategy="mean")
scaler = StandardScalerStrategy()

# 3. Construct Workflow Pipeline
pipeline.add_step("Impute Missing Values", lambda df: imputer.transform(df, ['age', 'salary']))
pipeline.add_step("Standardize Scaler Features", lambda df: scaler.fit_transform(df, ['age', 'salary']))
pipeline.add_step("Functional Exponential Scaling", lambda df: FunctionalTransformer.apply_math_mapping(df, 'age', lambda x: x ** 2))

# 4. Run Pipeline
clean_df = pipeline.execute(data)
print(clean_df)
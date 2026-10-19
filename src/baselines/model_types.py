from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelRun:
    model_id: str
    display_name: str
    prediction_scaled_log: np.ndarray
    prediction_heat_flow: np.ndarray
    selected_parameters: Dict[str, Any]
    complexity: Dict[str, Any]
    efficiency: Dict[str, Any]
    tuning_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)
    model_object: Optional[Any] = None

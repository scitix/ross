import sys
import time
from datetime import datetime
import pandas as pd
import subprocess
from typing import List, Dict, Any, Tuple
from pathlib import Path

from pareto.report import log_final_summary
import pareto.pareto as pa

pareto_frontier_result_df = {}
pareto_df_1 = pd.read_csv('log/pareto_1223_1608_Disagg.csv')
pareto_df_2 = pd.read_csv('log/pareto_1223_1655_Colocate.csv')

if not pareto_df_1.empty:
    pareto_frontier_df = pa.get_pareto_front(pareto_df_1, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
    print("\n=== disagg pareto ===")
    print(pareto_frontier_df)
    pareto_frontier_result_df["Disagg"] = pareto_frontier_df

if not pareto_df_2.empty:
    pareto_frontier_df = pa.get_pareto_front(pareto_df_2, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
    print("\n=== colocate pareto ===")
    print(pareto_frontier_df)
    pareto_frontier_result_df["Colocate"] = pareto_frontier_df

log_final_summary(pareto_frontier_result_df)
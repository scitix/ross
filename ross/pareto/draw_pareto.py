import sys
from typing import Dict
import pandas as pd

import pareto.pareto as pa
from pareto.report import log_final_summary


def draw_pareto_fronts(pareto_fronts: Dict[str, pd.DataFrame]) -> None:
    """Print a terminal Pareto-frontier plot for each mode in *pareto_fronts*.

    Args:
        pareto_fronts: mapping of mode label (e.g. "Colocate", "Disagg") to
                       a DataFrame that already contains the Pareto-front rows,
                       with at least "tokens/s/user" and "tokens/s/gpu" columns.
    """
    log_final_summary(pareto_fronts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Draw Pareto fronts from saved CSV files.")
    parser.add_argument("--disagg",   type=str, default="", metavar="CSV", help="Disagg pareto CSV")
    parser.add_argument("--colocate", type=str, default="", metavar="CSV", help="Colocate pareto CSV")
    args = parser.parse_args()

    pareto_frontier_result_df: Dict[str, pd.DataFrame] = {}

    if args.disagg:
        df = pd.read_csv(args.disagg)
        if not df.empty:
            pf = pa.get_pareto_front(df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
            print("\n=== disagg pareto ===")
            print(pf)
            pareto_frontier_result_df["Disagg"] = pf

    if args.colocate:
        df = pd.read_csv(args.colocate)
        if not df.empty:
            pf = pa.get_pareto_front(df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True).reset_index()
            print("\n=== colocate pareto ===")
            print(pf)
            pareto_frontier_result_df["Colocate"] = pf

    draw_pareto_fronts(pareto_frontier_result_df)

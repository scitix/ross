import pandas as pd
from pareto.pareto import draw_pareto_to_string

def log_final_summary(
    # best_throughputs: dict[str, float],
    # best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    # task_configs: dict[str, TaskConfig],
    # mode: str,
    pareto_x_axis: dict[str, str] | None = None,
):
    """Log final summary of configuration results"""

    summary_box = []

    # ============================= pareto frontier
    pareto_plot_buf = ""
    if len(pareto_fronts) <= 10:  # avoid overly crowded plots
        summary_box.append("  Pareto Frontier:")
        target_x_axis = "tokens/s/user"
        # if pareto_x_axis:
        #     target_x_axis = pareto_x_axis.get(chosen_exp, target_x_axis)
        series_payload = []
        for name, df in pareto_fronts.items():
            if df is None or df.empty:
                continue
            series_axis = pareto_x_axis.get(name, target_x_axis) if pareto_x_axis else target_x_axis
            if series_axis != target_x_axis:
                continue
            series_payload.append({"df": df, "label": name})
        highlight_series = None
        
        pareto_plot_buf = draw_pareto_to_string(
            f"Pareto Frontier",
            series_payload,
            highlight=highlight_series,
            x_label=target_x_axis,
            y_label="tokens/s/gpu",
        )
        summary_box.append(pareto_plot_buf)
    summary_box.append("  " + "-" * 76)
    # summary_box.append("*" * 80)
    print("\n" + "\n".join(summary_box))
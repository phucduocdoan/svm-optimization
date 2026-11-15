import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def test_heatmap_text_contrasts_with_dark_and_light_cells():
    spec = importlib.util.spec_from_file_location(
        "report_benchmark", Path(__file__).resolve().parents[1] / "experiments" / "06_report_benchmark.py"
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    figure = benchmark.plot_grid_heatmap(pd.DataFrame([[0.0, 1.0]]))
    try:
        annotations = figure.axes[0].texts
        assert [text.get_text() for text in annotations] == ['0.000', '1.000']
        assert [text.get_color() for text in annotations] == ['black', 'white']
    finally:
        plt.close(figure)

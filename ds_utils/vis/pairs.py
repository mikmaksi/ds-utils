import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_r_coef(x, y, **kwargs):
    ax = plt.gca()
    corr_coef = np.corrcoef(x, y)[0, 1]
    ax.annotate(
        f"Corr: {corr_coef:.2f}",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        fontsize="large",
        color="red",
    )


def plot_pairs(X: pd.DataFrame, color: pd.Series = None) -> sns.PairGrid:
    g = sns.PairGrid(X)
    g.map_diag(sns.histplot)
    g.map_lower(sns.scatterplot, hue=color)
    g.map_upper(plot_r_coef)
    return g

import numpy as np
import matplotlib.pyplot as plt
import mpltern

from Outputfile_Ingestion import parse_ternaryVLE_tab_blocks

# This code requires mpltern >= 1.0.0

def plot_ternary_heatmap_mpltern(df, x_cols=['x1', 'x2', 'x3'], value_col=None, title=None, cmap='viridis', vmin=None, vmax=None):
    """
    Plots a ternary heatmap from a DataFrame with columns x1, x2, x3 and a value column using mpltern.
    Args:
        df (pd.DataFrame): DataFrame with x1, x2, x3 columns (fractions summing to 1).
        x_cols (list): Names of the columns for the ternary axes.
        value_col (str): Name of the column to use for color.
        title (str): Plot title.
        cmap (str): Colormap for heatmap.
        vmin, vmax: Color scale limits.
    """
    # Prepare data
    t = df[x_cols[0]].values
    l = df[x_cols[1]].values
    r = df[x_cols[2]].values
    v = df[value_col].values
    # Normalize if not already
    s = t + l + r
    t = t / s
    l = l / s
    r = r / s
    # Plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection='ternary')
    cs = ax.tripcolor(t, l, r, v, cmap=cmap, vmin=vmin, vmax=vmax)
    # Set ternary axis labels using standard axis label methods
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    # Omit set_zlabel for linter compatibility
    if title:
        ax.set_title(title)
    # Colorbar
    cax = ax.inset_axes((1.05, 0.1, 0.05, 0.8), transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label(value_col, rotation=270, va='baseline')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df_real, df_nrtl_params, df_nrtl_pred = parse_ternaryVLE_tab_blocks(
        r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles\ternaryVLE_h2o_lacticacid_n-undecane.tab"
    )
    plot_ternary_heatmap_mpltern(
        df_real,
        value_col='y1',
        title='COSMOtherm Prediction: y1 Heatmap',
        cmap='viridis',
    )



    

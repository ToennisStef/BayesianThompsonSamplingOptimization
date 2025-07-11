import numpy as np
import matplotlib.pyplot as plt
import mpltern

import plotly
import plotly.graph_objects as go
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

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
    total = t + l + r
    t = t / total
    l = l / total
    r = r / total
    # Plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection='ternary')
    # cs = ax.tripcolor(t, l, r, v, cmap=cmap, vmin=vmin, vmax=vmax)
    cs = ax.tricontourf(t, l, r, v, cmap=cmap, vmin=vmin, vmax=vmax)
    # Set ternary axis labels using standard axis label methods
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    # Omit set_zlabel for linter compatibility
    if title:
        ax.set_title(title)
    # Colorbar
    cax = ax.inset_axes((1.05, 0.1, 0.05, 0.8), transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    if value_col:
        colorbar.set_label(value_col, rotation=270, va='baseline')
    plt.tight_layout()
    plt.show()

def plot_ternary_lle_tielines(df_lle, title=None, color='red', alpha=0.7, linewidth=2):
    """
    Plots LLE tie lines on a ternary diagram.
    Args:
        df_lle (pd.DataFrame): DataFrame with LLE data containing columns:
            x`(1), x`(2), x`(3) for phase 1 compositions
            x``(1), x``(2), x``(3) for phase 2 compositions
        title (str): Plot title.
        color (str): Color for tie lines.
        alpha (float): Transparency of tie lines.
        linewidth (float): Width of tie lines.
    """
    if df_lle is None or df_lle.empty:
        print("No LLE data to plot")
        return
    
    # Create figure
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, projection='ternary')
    
    # Plot tie lines
    for idx, row in df_lle.iterrows():
        # Skip rows with all zeros (no tie line)
        if (row['x`(1)'] == 0 and row['x`(2)'] == 0 and row['x`(3)'] == 0) or \
           (row['x``(1)'] == 0 and row['x``(2)'] == 0 and row['x``(3)'] == 0):
            continue
            
        # Phase 1 composition (x`)
        x1_1 = row['x`(1)']
        x2_1 = row['x`(2)']
        x3_1 = row['x`(3)']
        
        # Phase 2 composition (x``)
        x1_2 = row['x``(1)']
        x2_2 = row['x``(2)']
        x3_2 = row['x``(3)']
        
        # Normalize compositions
        sum1 = x1_1 + x2_1 + x3_1
        sum2 = x1_2 + x2_2 + x3_2
        
        if sum1 > 0 and sum2 > 0:
            x1_1, x2_1, x3_1 = x1_1/sum1, x2_1/sum1, x3_1/sum1
            x1_2, x2_2, x3_2 = x1_2/sum2, x2_2/sum2, x3_2/sum2
            
            # Plot tie line
            ax.plot([x1_1, x1_2], [x2_1, x2_2], [x3_1, x3_2], 
                   color=color, alpha=alpha, linewidth=linewidth)
            
            # Plot phase points
            ax.scatter([x1_1, x1_2], [x2_1, x2_2], [x3_1, x3_2], 
                      color=color, s=30, alpha=alpha)
    
    # Set labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()

def plot_ternary_vle_with_lle(df_vle, df_lle, value_col='mu1+RTln(x1)', title=None, 
                             cmap='viridis', vmin=None, vmax=None, 
                             lle_color='red', lle_alpha=0.7, lle_linewidth=2):
    """
    Plots VLE heatmap with LLE tie lines overlaid.
    Args:
        df_vle (pd.DataFrame): VLE data for heatmap.
        df_lle (pd.DataFrame): LLE data for tie lines.
        value_col (str): Column to use for VLE color mapping.
        title (str): Plot title.
        cmap (str): Colormap for VLE heatmap.
        vmin, vmax: Color scale limits for VLE.
        lle_color (str): Color for LLE tie lines.
        lle_alpha (float): Transparency of LLE tie lines.
        lle_linewidth (float): Width of LLE tie lines.
    """
    if df_vle is None or df_vle.empty:
        print("No VLE data to plot")
        return
    
    # Prepare VLE data
    t = df_vle['x1'].values
    l = df_vle['x2'].values
    r = df_vle['x3'].values
    v = df_vle[value_col].values
    
    # Normalize if not already
    total = t + l + r
    t = t / total
    l = l / total
    r = r / total
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection='ternary')
    
    # Plot VLE heatmap
    cs = ax.tricontourf(t, l, r, v, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Plot LLE tie lines if available
    if df_lle is not None and not df_lle.empty:
        for idx, row in df_lle.iterrows():
            # Skip rows with all zeros (no tie line)
            if (row['x`(1)'] == 0 and row['x`(2)'] == 0 and row['x`(3)'] == 0) or \
               (row['x``(1)'] == 0 and row['x``(2)'] == 0 and row['x``(3)'] == 0):
                continue
                
            # Phase 1 composition (x`)
            x1_1 = row['x`(1)']
            x2_1 = row['x`(2)']
            x3_1 = row['x`(3)']
            
            # Phase 2 composition (x``)
            x1_2 = row['x``(1)']
            x2_2 = row['x``(2)']
            x3_2 = row['x``(3)']
            
            # Normalize compositions
            sum1 = x1_1 + x2_1 + x3_1
            sum2 = x1_2 + x2_2 + x3_2
            
            if sum1 > 0 and sum2 > 0:
                x1_1, x2_1, x3_1 = x1_1/sum1, x2_1/sum1, x3_1/sum1
                x1_2, x2_2, x3_2 = x1_2/sum2, x2_2/sum2, x3_2/sum2
                
                # Plot tie line
                ax.plot([x1_1, x1_2], [x2_1, x2_2], [x3_1, x3_2], 
                       color=lle_color, alpha=lle_alpha, linewidth=lle_linewidth)
                
                # Plot phase points
                ax.scatter([x1_1, x1_2], [x2_1, x2_2], [x3_1, x3_2], 
                          color=lle_color, s=30, alpha=lle_alpha)
    
    # Set labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if title:
        ax.set_title(title)
    
    # Colorbar
    cax = ax.inset_axes((1.05, 0.1, 0.05, 0.8), transform=ax.transAxes)
    colorbar = fig.colorbar(cs, cax=cax)
    if value_col:
        colorbar.set_label(value_col, rotation=270, va='baseline')
    
    plt.tight_layout()
    plt.show()


# (Keep your transform_simplex_to_xy_plane function as it is)
def transform_simplex_to_xy_plane(x, y, z):
    """
    Transforms simplex coordinates (x, y, z) to 2D Cartesian coordinates.
    Args:
        x, y, z: Coordinates in the simplex.
    Returns:
        Tuple of (x_cartesian, y_cartesian).
    """
    def rotaion_matrix(theta, vector):
        """Returns a rotation matrix for a given angle theta around a vector."""
        c, s = np.cos(theta), np.sin(theta)
        x, y, z = vector
        return np.array([[c + (1 - c) * x * x, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
                         [(1 - c) * y * x + s * z, c + (1 - c) * y * y, (1 - c) * y * z - s * x],
                         [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]])

    rot = rotaion_matrix(np.pi / 4, (1, -1, 0))
    vec = np.array([x, y, z])
    vec_rotated = rot @ vec
    x_rot, y_rot, z_rot = vec_rotated
    return x_rot, y_rot

def interp_2dsimplex_to_mgrid(xp, yp, z_values, grid_res=500j):
    """
    Interpolates the 2D simplex coordinates to a grid for surface plotting.
    Args:
        xp, yp: 1D arrays of x and y coordinates in the simplex.
        z_values: 1D array of z values corresponding to (xp, yp).
    Returns:
        grid_x, grid_y, grid_z: 2D arrays for plotting.
    """
    grid_x, grid_y = np.mgrid[xp.min():xp.max():grid_res, yp.min():yp.max():grid_res]

    grid_z = griddata(np.vstack((xp, yp)).T, z_values, (grid_x, grid_y), method='linear')

    # The Delaunay triangulation finds the convex hull of your points.
    # `find_simplex` will return -1 for any grid point outside the hull.
    tri = Delaunay(np.vstack((xp, yp)).T)
    mask = tri.find_simplex(np.vstack((grid_x.ravel(), grid_y.ravel())).T) < 0
    grid_z.ravel()[mask] = np.nan # Set points outside the hull to NaN
    return grid_x, grid_y, grid_z

def plot_3d_ternary_with_lle(df_real, df_lle=None, 
                             value_col='G_total', title=None, cmap='viridis'):

    # --- 2. Project Data to 2D Cartesian Plane ---
    xp, yp = transform_simplex_to_xy_plane(df_real['x1'], df_real['x2'], df_real['x3'])
    z_values = df_real[value_col]
    z_min = z_values.min()
    z_max = z_values.max()

    corners = np.array([
        [1, 0, 0],  
        [0, 1, 0],  
        [0, 0, 1],
        [1, 0, 0],
    ])

    corner_df = pd.DataFrame(corners, columns=['x1', 'x2', 'x3'])
    corner_df["name"] = ['A', 'B', 'C', 'A']  # Repeat first corner to close the triangle
    xp_corners, yp_corners = transform_simplex_to_xy_plane(corners[:, 0], corners[:, 1], corners[:, 2])


    grid_x, grid_y, grid_z = interp_2dsimplex_to_mgrid(xp, yp, z_values)

    fig = go.Figure(data=[go.Surface(
        x=grid_x[:, 0], # X coordinates for the grid
        y=grid_y[0, :], # Y coordinates for the grid
        z=grid_z,       # Z values (G_total) on the grid
        colorscale='Viridis',
        colorbar=dict(title='Gibbs Free Energy (G_total)'),
        cmin=z_values.min(),
        cmax=z_values.max(),
    )])

    # Connect the corners of the simplex
    fig.add_trace(go.Scatter3d(
        x=xp_corners,
        y=yp_corners,
        z=np.ones_like(xp_corners) * z_min,
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=1, color='black'),
        hoverinfo='text',
        hovertext=corner_df['name'].tolist()
    ))

    fig.add_trace(go.Scatter3d(
        x=xp_corners,
        y=yp_corners,
        z=np.ones_like(xp_corners) * z_max,
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(size=1, color='black'),
    ))
    

    # Plot LLE tie lines if available
    if df_lle is not None and not df_lle.empty:
        df_lle_clean = df_lle[~((df_lle['x`(1)'] == 0) & (df_lle['x`(2)'] == 0) & (df_lle['x`(3)'] == 0) &
                             (df_lle['x``(1)'] == 0) & (df_lle['x``(2)'] == 0) & (df_lle['x``(3)'] == 0))].reset_index()

        x1_1, x2_1 = transform_simplex_to_xy_plane(df_lle_clean['x`(1)'], df_lle_clean['x`(2)'], df_lle_clean['x`(3)'])
        x1_2, x2_2 = transform_simplex_to_xy_plane(df_lle_clean['x``(1)'], df_lle_clean['x``(2)'], df_lle_clean['x``(3)'])

        for idx, row in df_lle_clean.iterrows():
            # Add LLE tie line
            fig.add_trace(go.Scatter3d(
                x=[x1_1[idx], x1_2[idx]],
                y=[x2_1[idx], x2_2[idx]],
                z=[z_min, z_min],  # Set Z to a constant for the tie line
                mode='lines+markers',
                line=dict(color='teal', width=3),
                marker=dict(size=3, color='teal'),
                name='LLE Tie Line'
            ))

        # Connect the LLE border points with lines
        fig.add_trace(go.Scatter3d(
            x=x1_1[np.argsort(df_lle_clean['x`(2)'])],
            y=x2_1[np.argsort(df_lle_clean['x`(2)'])],
            z=z_min * np.ones_like(x1_1),  # Set Z to a constant for the points
            mode='lines+markers',
            line=dict(color='teal', width=3),
            marker=dict(size=3, color='teal'),
            name='LLE Tie Line'
        ))

        # Connect the LLE border points with lines
        fig.add_trace(go.Scatter3d(
            x=x1_2[np.argsort(df_lle_clean['x``(2)'])],
            y=x2_2[np.argsort(df_lle_clean['x``(2)'])],
            z=z_min * np.ones_like(x1_2),  #
            mode='lines+markers',
            line=dict(color='teal', width=3),
            marker=dict(size=3, color='teal'),
            name='LLE Tie Line'
        ))
            
    # Plot values on the surface
    fig.update_layout(
        title=title if title else f'Ternary Plot of {value_col}',
        scene=dict(
            xaxis_title='Projected X',
            yaxis_title='Projected Y',
            zaxis_title=value_col,
            # Aspect ratio can be adjusted for better visualization
            aspectratio=dict(x=1, y=1, z=0.7)
        ),
        width=1920,
        height=800,
        template='plotly_white',
    )

#center in page (currently is left aligned)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1)
        )
    )

    fig.show()

if __name__ == "__main__":
    # --- 1. Load and Prepare Data ---
    
    df_real, df_lle, df_lle_renom, df_nrtl_params, df_nrtl_pred, df_lle_nrtl, df_lle_nrtl_renom = parse_ternaryVLE_tab_blocks(
        # r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles\ternaryVLE_h2o_lacticacid_n-undecane.tab"
        r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles\random_lle2.tab"
    )

    df_real["G_ideal"] = df_real["x1"] * np.log(df_real["x1"] + 1e-9) \
                       + df_real["x2"] * np.log(df_real["x2"] + 1e-9) \
                       + df_real["x3"] * np.log(df_real["x3"] + 1e-9)
    
    df_real["G_total"] = df_real["G_ideal"] + df_real["G^E"]

    plot_3d_ternary_with_lle(
        df_real,
        df_lle,
        value_col='G_total',
        title='COSMOtherm Prediction: G_total Heatmap',
        cmap='viridis'
    )

    # plot_ternary_heatmap_mpltern(
    #     df_real,
    #     value_col='ln(gamma1)',
    #     title='COSMOtherm Prediction: y1 Heatmap',
    #     cmap='viridis',
    # )

    # plot_ternary_lle_tielines(
    #     df_lle,
    #     title='COSMOtherm Prediction: y1 Heatmap',
    #     color='red',
    #     alpha=0.7,
    #     linewidth=2
    # )

    # plot_ternary_vle_with_lle(
    #     df_real,
    #     df_lle,
    #     value_col='G_total',
    #     title='',
    #     cmap='viridis',
    # )
    
    # plot_ternary_heatmap_mpltern(
    #     df_real,
    #     value_col='mu1+RTln(x1)',
    #     title='COSMOtherm Prediction: y1 Heatmap',
    #     cmap='viridis',
    # )


    # plot_ternary_heatmap_mpltern(
    #     df_real,
    #     value_col='ptot',
    #     title='COSMOtherm Prediction: y1 Heatmap',
    #     cmap='viridis',
    # )


    

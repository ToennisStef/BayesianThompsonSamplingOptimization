import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def parity_plot(y_true, y_pred, title="", save_path=None):
    """
    Create a parity plot (y_true vs y_pred) with a y=x reference line.
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
        title: optional plot title
        save_path: if provided, save the plot to this path
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    plt.title(f"Parity Plot\nMSE: {mse:.4f}, R2: {r2:.4f}\n{title}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show() 



# Convert to lists and save as json
def save_to_json(np_array_list, filename):
    import json
    
    if type(np_array_list) is not list:
        np_array_list = np_array_list.tolist()
    else:
        np_array_list = [arr.tolist() for arr in np_array_list]
    with open(f"{filename}.json", 'w') as f:
        json.dump(np_array_list, f)
        
def load_from_json(filename):
    import json
    with open(f"{filename}.json", 'r') as f:
        data = json.load(f)
    return [np.array(arr) for arr in data]
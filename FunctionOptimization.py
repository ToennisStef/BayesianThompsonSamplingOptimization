from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
import torch
from scipy.optimize import shgo
import matplotlib.pyplot as plt


def main():
    """
    Main function to perform optimization on multiple functions using the SHGO algorithm and plot the results.
    The function performs the following steps:
    1. Initializes an instance of the MF1 class.
    2. Optimizes four functions (f1, f2, f3, f4) using the SHGO algorithm with specified bounds and options.
    3. Prints the optimization results for each function.
    4. Prints the optimal x values for each function.
    5. Generates a range of x values and computes the corresponding y values for each function.
    6. Plots the functions and their optimal points using matplotlib.
    Note:
    - The MF1 class and its methods (f1, f2, f3, f4) should be defined elsewhere in the code.
    - The SHGO algorithm is used for optimization.
    - The bounds for the optimization are obtained from the MF1 instance.
    - The results are plotted using matplotlib.
    Returns:
        None
    """


    mf1 = MF1()
    
    result_f1 = shgo(mf1.f1, bounds=mf1.bounds.T.numpy(), options={'to_numpy': True}, n=100, iters=100)
    result_f2 = shgo(mf1.f2, bounds=mf1.bounds.T.numpy(), options={'to_numpy': True}, n=100, iters=100)
    result_f3 = shgo(mf1.f3, bounds=mf1.bounds.T.numpy(), options={'to_numpy': True}, n=100, iters=100)
    result_f4 = shgo(mf1.f4, bounds=mf1.bounds.T.numpy(), options={'to_numpy': True}, n=100, iters=100)
    
    print("Optimization result for f1:", result_f1)
    print("Optimization result for f2:", result_f2)
    print("Optimization result for f3:", result_f3)
    print("Optimization result for f4:", result_f4)
    
    print("Optimal x for f1: {:.15f}".format(result_f1.x[0]))
    print("Optimal x for f2: {:.15f}".format(result_f2.x[0]))
    print("Optimal x for f3: {:.15f}".format(result_f3.x[0]))
    print("Optimal x for f4: {:.15f}".format(result_f4.x[0]))
    
    x1 = torch.linspace(mf1.lower_bound.item(), mf1.upper_bound.item(), 1000)
    y1 = mf1.f1(x1)
    y2 = mf1.f2(x1)
    y3 = mf1.f3(x1)
    y4 = mf1.f4(x1)
    
    fig, ax = plt.subplots()
    
    ax.plot(x1.numpy(), y1.numpy(), label='f1', color='red')
    ax.plot(x1.numpy(), y2.numpy(), label='f2', color='green')
    ax.plot(x1.numpy(), y3.numpy(), label='f3', color='blue')
    ax.plot(x1.numpy(), y4.numpy(), label='f4', color='purple')
    
    ax.scatter(result_f1.x, result_f1.fun, color='red', label='Optimum f1')
    ax.scatter(result_f2.x, result_f2.fun, color='green', label='Optimum f2')
    ax.scatter(result_f3.x, result_f3.fun, color='blue', label='Optimum f3')
    ax.scatter(result_f4.x, result_f4.fun, color='purple', label='Optimum f4')
    
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Function Optimization Results')
    
    plt.show()
    
if __name__ == "__main__":
    main()
from .Testfunctions import TestFunction
import torch
import numpy as np
import warnings

# Goal of Mulit-Fidelity Optimization is to find the optimal value 
# of the function of the highest fidelity level.

class MFTestFunction(TestFunction):
    """
    Base class for multi-fidelity test functions
    
    """
    def __init__(self, 
                 dim: int, 
                 shift=None,
                 rotation=None,
                 dtype=torch.float64):
        super().__init__(
                        dim=dim, 
                        shift=shift, 
                        rotation=rotation,
                        dtype=dtype
                 )
        self._n_fidelities = None # Number of fidelity levels
        self._fidelities = None # List of fidelity levels
        self._optimal_positions = None # Optimal positions for each fidelity level
        self._optimal_values = None # Optimal values for each fidelity level
        self._funcs = None # List of functions for each fidelity level
        self._funcs_neg = None # List of negated functions for each fidelity level
        
    @property
    def n_fidelities(self):
        """
        Returns the number of fidelity levels.

        Returns:
        int: The number of fidelity levels.
        """
        return self._n_fidelities
    
    @property
    def fidelities(self):
        """
        Returns the list of fidelity levels.

        Returns:
        list: The list of fidelity levels.
        """
        return self._fidelities
        
    @property
    def optimal_positions(self):
        """
        Returns the optimal positions for each fidelity level.

        Returns:
        torch.tensor: The optimal positions for each fidelity level.
        """
        return self._optimal_positions
    
    @property
    def optimal_values(self):
        """
        Returns the optimal values for each fidelity level.

        Returns:
        torch.tensor: The optimal values for each fidelity level.
        """
        return self._optimal_values
    
    @property
    def funcs(self):
        """
        Returns the list of functions for each fidelity level.

        Returns:
        list: The list of functions for each fidelity level.
        """
        return self._funcs
    
    @property
    def funcs_neg(self):
        """
        Returns the list of negated functions for each fidelity level.

        Returns:
        list: The list of negated functions for each fidelity level.
        """
        return self._funcs_neg
        
    def __call__(self, x, fidelity):
        """
        Placeholder for the multi-fidelity test function. Overwirtes the TestFunction class method.
        """
        raise NotImplementedError("The function has not been implemented yet.")

    def neg(self, x, fidelity):
        """
        Negates the function value. Overwirtes the TestFunction class method.
        """
        return -self.__call__(x, fidelity)

    @staticmethod
    def check_dtype(func):
        def wrapper(self, x, *args, **kwargs):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=self.dtype)
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input must be a numpy.ndarray or torch.Tensor, but got {type(x)}")
            if x.dtype != self.dtype:
                warnings.warn(f"dtype mismatch! Input tensor has dtype {x.dtype}, but expected {self.dtype}", UserWarning)
            return func(self, x, *args, **kwargs)
        return wrapper
    

class MF1(MFTestFunction):
    """
    Multi-Fidelity adaptation of the Forrester function.
    This class represents a multi-fidelity test function with 4 discrete fidelity levels.
    Attributes:
        _name (str): Name of the test function.
        _n_fidelities (list): Number of fidelity levels.
        _fidelities (torch.Tensor): Tensor containing the fidelity levels.
        _lower_bound (torch.Tensor): Lower bound of the input domain.
        _upper_bound (torch.Tensor): Upper bound of the input domain.
        _bounds (torch.Tensor): Bounds of the input domain.
        _optimal_position (torch.Tensor): Optimal position in the input domain.
        _optimal_value (torch.Tensor): Optimal value of the function.
    Methods:
        __call__(x, fidelity):
            Evaluate the function at a given input `x` and fidelity level.
        f1(x):
            High-fidelity function.
            Equation: (6x - 2)^2 * sin(12x - 4)
        f2(x):
            Medium-high fidelity function.
            Equation: (5.5x - 2.5)^2 * sin(12x - 4)
        f3(x):
            Medium-low fidelity function.
            Equation: 0.75 * f1(x) + 5 * (x - 0.5) - 2
        f4(x):
            Low-fidelity function.
            Equation: 0.5 * f1(x) + 10 * (x - 0.5) - 5
    """
    def __init__(self, 
                 shift:torch.Tensor=None, 
                 rotation:torch.Tensor=None, 
                 dtype:torch.dtype=torch.float64):
        super().__init__(dim=1,shift=shift,rotation=rotation,dtype=dtype)
        self._name = "MF1"
        self._n_fidelities = 4
        self._fidelities = [0, 1, 2, 3]
        self._lower_bound = torch.tensor([0.0] * self.dim, dtype=dtype)
        self._upper_bound = torch.tensor([1.0] * self.dim, dtype=dtype)
        self._bounds = torch.vstack([self.lower_bound, self.upper_bound])
        self._optimal_position = torch.tensor([0.757248750905720], dtype=dtype)
        self._optimal_value = torch.tensor([-6.020740055767057], dtype=dtype)
        self._optimal_positions = {
            0: torch.tensor([0.092392869763708], dtype=dtype),
            1: torch.tensor([0.750824446226849], dtype=dtype),
            2: torch.tensor([0.766880460664725], dtype=dtype),
            3: torch.tensor([0.757248750905720], dtype=dtype),
        }
        self._optimal_values = {
            0: torch.tensor(-9.334904876953633, dtype=dtype),
            1: torch.tensor(-5.245213382473921, dtype=dtype),
            2: torch.tensor(-2.6035040085576537, dtype=dtype),
            3: torch.tensor(-6.020740055767057, dtype=dtype)
        }
        self._funcs = {f: lambda x, fidelity=f: self.__call__(x, fidelity) for f in self.fidelities} # Dictionary of functions for each fidelity level
        self._funcs_neg = {f: lambda x, fidelity=f: self.neg(x, fidelity) for f in self.fidelities} # Dictionary of negated functions for each fidelity level
    
    @MFTestFunction.check_dtype
    def __call__(self, 
                 x:torch.tensor, 
                 fidelity:int,
                 to_numpy:bool=False):
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        if fidelity == 3:
            return self.f1(x, to_numpy=to_numpy)
        elif fidelity == 2:
            return self.f2(x, to_numpy=to_numpy)
        elif fidelity == 1:
            return self.f3(x, to_numpy=to_numpy)
        elif fidelity == 0:
            return self.f4(x, to_numpy=to_numpy) 
        else:
            raise ValueError("Fidelity level not supported")
        
    @MFTestFunction.check_dtype
    def f1(self, x:torch.Tensor, to_numpy:bool=False):
        if to_numpy:
            return ((6*x - 2)**2 * torch.sin(12*x - 4)).numpy()
        return (6*x - 2)**2 * torch.sin(12*x - 4)
    
    @MFTestFunction.check_dtype
    def f2(self, x:torch.Tensor, to_numpy:bool=False):
        if to_numpy:
            return ((5.5*x - 2.5)**2 * torch.sin(12*x - 4)).numpy()
        return (5.5*x - 2.5)**2 * torch.sin(12*x - 4)
    
    @MFTestFunction.check_dtype
    def f3(self, x:torch.Tensor, to_numpy:bool=False):
        if to_numpy:
            return (0.75*self.f1(x) + 5*(x-0.5) - 2).numpy()
        return 0.75*self.f1(x) + 5*(x-0.5) - 2
    
    @MFTestFunction.check_dtype
    def f4(self, x:torch.Tensor, to_numpy:bool=False):
        if to_numpy:
            return (0.5*self.f1(x) + 10*(x-0.5) - 5).numpy()
        return 0.5*self.f1(x) + 10*(x-0.5) - 5
    
    
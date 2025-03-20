import math
import torch
import numpy as np


class TestFunction:
    def __init__(self, 
                dim:int=1,
                shift:torch.tensor=None,
                rotation:torch.tensor=None,
                dtype:torch.dtype=torch.float64):
        self._name = None
        self._dim = dim
        self._optimal_value = None
        self._optimal_position = None
        self._lower_bound = None
        self._upper_bound = None
        self._bounds = None # Bounds of the input space
        self._shift = shift
        self._rotation = rotation
        self._dtype = dtype
        
    @property
    def name(self):
        """
        Returns the name of the test function.

        Returns:
        str: The name of the test function.
        """
        return self._name
    
    @property
    def dim(self):
        """
        Returns the input dimension of the test function.

        Returns:
        int: The input dimension of the test function.
        """
        return self._dim
    
    @property
    def dtype(self):
        """
        Returns the data type of the test function.

        Returns:
        torch.dtype: The data type of the test function.
        """
        return self._dtype
    
    @property
    def optimal_value(self):
        """
        Returns the optimal value of the test function.

        Returns:
        float: The optimal value of the test function.
        """
        return self._optimal_value
    
    @property
    def optimal_position(self):
        """
        Returns the optimal position of the test function.

        Returns:
        torch.Tensor: The optimal position of the test function.
        """
        return self._optimal_position
    
    @property
    def lower_bound(self):
        """
        Returns the lower bound of the input space.

        Returns:
        torch.Tensor: The lower bound of the input space.
        """
        return self._lower_bound
    
    @property
    def upper_bound(self):
        """
        Returns the upper bound of the input space.

        Returns:
        torch.Tensor: The upper bound of the input space.
        """
        return self._upper_bound
    
    @property
    def bounds(self):
        """
        Returns the bounds of the input space.

        Returns:
        torch.Tensor: The bounds of the input space.
        """
        return self._bounds
    
    @property
    def shift(self):
        """
        Returns the shift of the test function.

        Returns:
        torch.Tensor: The shift of the test function.
        """
        return self._shift
    
    @property
    def rotation(self):
        """
        Returns the rotation of the test function.

        Returns:
        torch.Tensor: The rotation of the test function.
        """
        return self._rotation
    
    @lower_bound.setter
    def lower_bound(self, value):
        assert isinstance(value, torch.Tensor), f"lower_bound must be a torch.Tensor object"
        assert value.shape == (self.dim,), f"lower_bound must have the same dimension as the test function."
        self._lower_bound = value.to(self.dtype)
        self._update_bounds()

    @upper_bound.setter
    def upper_bound(self, value):
        assert isinstance(value, torch.Tensor), f"upper_bound must be a torch.Tensor object."
        assert value.shape == (self.dim,), f"upper_bound must have the same dimension as the test function."
        self._upper_bound = value.to(self.dtype)
        self._update_bounds()

    @bounds.setter
    def bounds(self, value):
        assert isinstance(value, torch.Tensor), f"bounds must be a torch.Tensor object."
        assert value.shape == (2, self.dim), f"bounds must have the shape (2, {self.dim})."
        self._bounds = value.to(self.dtype)
        self._lower_bound = value[0]
        self._upper_bound = value[1]

    def _update_bounds(self):
        if self._lower_bound is not None and self._upper_bound is not None:
            self._bounds = torch.stack([self._lower_bound, self._upper_bound])

    @shift.setter
    def shift(self, value):
        assert isinstance(value, torch.Tensor), f"shift must be a torch.Tensor object."
        assert value.shape == (self.dim,), f"shift must have the same dimension as the test function."
        self._shift = value

    @rotation.setter
    def rotation(self, value):
        assert isinstance(value, torch.Tensor), f"rotation must be a torch.Tensor object."
        assert value.shape == (self.dim, self.dim), f"rotation must have the shape ({self.dim}, {self.dim})."
        self._rotation = value
        
    @dtype.setter
    def dtype(self, value):
        assert isinstance(value, torch.dtype), "dtype must be a torch.dtype object."
        self._dtype = value
    
    def __call__(self, x):
        """
        Placeholder for the test function.
        """
        raise NotImplementedError("Test function not implemented.")
    
    def neg(self, x):
        """
        Returns the negative value of the test function.

        Args:
        x (torch.Tensor): The input tensor for the test function.

        Returns:
        torch.Tensor: The negative value of the test function.
        """
        return -self.__call__(x)
    
    def numpy(self, x):
        return self.__call__(torch.tensor(x)).numpy()

    def neg_numpy(self, x):
        return self.neg(torch.tensor(x)).numpy()
    
    
    def __repr__(self):
        return f"{self.name}({self.dim})"
    
    def __str__(self):
        return f"{self.name}({self.dim})"
    
    def __eq__(self, other):
        if not isinstance(other, TestFunction):
            return False
        return self.name == other.name and self.dim == other.dim


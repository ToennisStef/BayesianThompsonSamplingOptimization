# filepath: [Tester.py](http://_vscodecontentref_/4)
from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
import torch

"""
Tester for the MF1 class.
"""

# Example test code
def test_name_mf1():
    mf1 = MF1()
    assert mf1.name == "MF1", f"Expected name MF1 but got {mf1.name}"

def test_dim_mf1():
    mf1 = MF1()
    assert mf1.dim == 1, f"Expected dim 1 but got {mf1.dim}"
    
def test_dtype_mf1():
    mf1 = MF1()
    assert mf1.dtype == torch.float64, f"Expected dtype torch.float64 but got {mf1.dtype}"
    
def test_optimal_value_mf1():
    mf1 = MF1()
    assert torch.allclose(mf1.optimal_value, torch.tensor([-6.020740], dtype=torch.float64)), f"Expected optimal value -6.020740 but got {mf1.optimal_value}"
    
def test_optimal_position_mf1():
    mf1 = MF1()
    assert torch.allclose(mf1.optimal_position, torch.tensor([0.75724876],dtype=torch.float64)), f"Expected optimal position [0.75724876] but got {mf1.optimal_position}"

def test_lower_bound_mf1():
    mf1 = MF1()
    assert torch.allclose(mf1.lower_bound, torch.tensor([0.0],dtype=torch.float64)), f"Expected lower bound [0.0] but got {mf1.lower_bound}"
    assert mf1.lower_bound.shape == torch.Size([1]), f"Expected lower bound shape [1] but got {mf1.lower_bound.shape}"
    assert mf1.lower_bound.shape == (1,), f"Expected lower bound shape [1] but got {mf1.lower_bound.shape}"
    assert mf1.lower_bound.dtype == mf1.dtype, f"Expected lower bound dtype {mf1.dtype} but got {mf1.lower_bound.dtype}"
    
def test_upper_bound_mf1():
    mf1 = MF1()
    assert torch.allclose(mf1.upper_bound, torch.tensor([1.0],dtype=torch.float64)), f"Expected upper bound [1.0] but got {mf1.upper_bound}"
    assert mf1.upper_bound.shape == torch.Size([1]), f"Expected upper bound shape [1] but got {mf1.upper_bound.shape}"
    assert mf1.upper_bound.shape == (1,), f"Expected upper bound shape [1] but got {mf1.upper_bound.shape}"
    assert mf1.upper_bound.dtype == mf1.dtype, f"Expected upper bound dtype {mf1.dtype} but got {mf1.upper_bound.dtype}"
    
def test_call_mf1():
    test_x = torch.tensor([0.5],dtype=torch.float64)
    mf1 = MF1()
    assert torch.allclose(mf1(test_x,3), mf1.f1(test_x)), f"Expected {mf1.f1(test_x)} but got {mf1(test_x,3)}"
    assert torch.allclose(mf1(test_x,2), mf1.f2(test_x)), f"Expected {mf1.f2(test_x)} but got {mf1(test_x,2)}"
    assert torch.allclose(mf1(test_x,1), mf1.f3(test_x)), f"Expected {mf1.f3(test_x)} but got {mf1(test_x,1)}"
    assert torch.allclose(mf1(test_x,0), mf1.f4(test_x)), f"Expected {mf1.f4(test_x)} but got {mf1(test_x,0)}"
    
if __name__ == "__main__":
    test_name_mf1()
    test_dim_mf1()
    test_dtype_mf1()
    test_optimal_position_mf1()
    test_optimal_value_mf1()
    test_lower_bound_mf1()
    test_upper_bound_mf1()
    test_call_mf1()
    print("All tests passed.")
################################################################################
#                             Attention Tests 
################################################################################

import torch
import pytest

from src.attention import SingleAttention, MultiHeadAttention

def use_model(kind): 

    if kind == "single": 
        return SingleAttention(n_embd=16,
            head_size=4,
            block_size=8,
            dropout=0.0)

    if kind == "multi": 
        return MultiHeadAttention(n_embd= 16, 
                                  num_heads= 4, 
                                  block_size= 8,
                                  dropout= 0.0)


# Test 1 : Output shapes
@pytest.mark.parametrize("kind,expected_dim", [
    ("single", 4),
    ("multi", 16),
])

def test_output_shape(kind, expected_dim):

    model = use_model(kind)

    x = torch.randn(2, 5, 16)

    output = model(x)

    return output.shape == (2,5, expected_dim)


# Test 2 : Causal mask 

@pytest.mark.parametrize("kind", ["single", "multi"])

def test_causality(kind): 
    torch.manual_seed(42)

    model = use_model(kind)

    model.eval()

    x = torch.randn(2, 5, 16)
    x_modified = x.clone()

    x_modified[:,3:, :] = torch.randn(2, 2, 16)

    with torch.no_grad(): 
        y_1 = model(x)
        y_2 = model(x_modified)

    assert torch.allclose(y_1[:,:3,:], y_2[:,:3,:], atol=1e-6)


# Test 3: gradient computations 

@pytest.mark.parametrize("kind", ["single", "multi"])
def test_gradients(kind): 

    model = use_model(kind)

    x = torch.randn(2, 5, 16, requires_grad=True)

    output = model(x)

    assert torch.isfinite(output).all()

    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None 
    assert torch.isfinite(x.grad).all()

    for parameter in model.parameters(): 
        assert parameter.grad is not None 
        assert torch.isfinite(parameter.grad).all()

        






















    

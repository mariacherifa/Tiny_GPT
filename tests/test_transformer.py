################################################################################
#                             Transformer Tests 
################################################################################

import torch
import pytest
from src.transformer import BlockTransformer


def use_model(n_embd): 
    return BlockTransformer(
        n_embd = n_embd, num_heads = 4, block_size = 8, dropout = 0.0
    )
# Test 1 : shapes test 
@pytest.mark.parametrize("n_embd", [4, 16])

def test_shapes(n_embd): 

    model = use_model(n_embd)

    x = torch.randn(2,2,n_embd)

    output = model(x)

    assert output.shape == x.shape

# test 2 : Causality 
@pytest.mark.parametrize("n_embd", [16])

def test_causality(n_embd): 

    model = use_model(n_embd)
    model.eval()

    x = torch.randn(2,5,n_embd)
    x_clone = x.clone()

    x_clone[:,3:,:] = torch.randn(2,2,n_embd)

    with torch.no_grad(): 
        y1 = model(x)
        y2 = model(x_clone)

    assert torch.allclose(y1[:,:3,:], y2[:,:3,:], atol = 1e-6)


# Test 3 : gradients computations 
@pytest.mark.parametrize("n_embd", [16])

def test_gradients(n_embd): 

    model = use_model(n_embd)

    x = torch.randn(2,5,n_embd, requires_grad=True)

    output = model(x)

    assert torch.isfinite(output).all()

    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None 
    assert torch.isfinite(x.grad).all()

    for paramter in model.parameters(): 

        assert paramter.grad is not None 
        assert torch.isfinite(paramter.grad).all()

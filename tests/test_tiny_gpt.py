################################################################################
#                             TinyGPT tests
################################################################################

import torch 
import pytest 
from src.model import TinyGPT

def use_model(vocab_size): 

    return TinyGPT(n_embd =64, vocab_size=vocab_size, 
                   num_heads= 4, block_size=64 ,
                   num_layers= 2, dropout = 0.1)


# Test 1 : shapes are matching 
@pytest.mark.parametrize("vocab_size", [65,100])

def test_shapes(vocab_size):
    model = use_model(vocab_size)

    x = torch.randint(0, vocab_size, (2, 10))

    _, logits = model(x)

    assert logits.shape == (2,10,vocab_size)


# Test 2 : different valid sequence length up to block size 
@pytest.mark.parametrize("sequence_len, vocab_size", [
    (30, 65),
    (63, 65)
])

def test_sequence_len(sequence_len,vocab_size):
    model = use_model(vocab_size)

    x =  torch.randint(0, vocab_size, (2, sequence_len))

    _, logits = model(x)

    assert logits.shape == (2,sequence_len,vocab_size)

# Test 3 : Causality test 

@pytest.mark.parametrize("vocab_size", [100])

def test_causality(vocab_size): 
    model = use_model(vocab_size)

    model.eval()

    x =  torch.randint(0, vocab_size, (2, 10))
    x_clone = x.clone()

    x_clone[:,5:] = torch.randint(0, vocab_size, (2, 5))

    with torch.no_grad(): 
        _,y1 = model(x)
        _,y2 = model(x_clone)

    assert torch.allclose(y1[:,:5,:], y2[:,:5,:], atol = 1e-6)


# Test 4: back propagation 
@pytest.mark.parametrize("vocab_size", [100])

def test_gradients(vocab_size): 
    model = use_model(vocab_size)

    x = torch.randint(0, vocab_size, (2, 11))
    inputs = x[:, :-1]
    targets = x[:, 1:]

    loss, logits = model(inputs, targets)

    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits).all()
    assert loss is not None 
    assert logits is not None 

    loss.backward()


    for name, parameter in model.named_parameters():

        assert parameter.grad is not None, \
            f"No gradient for {name}"

        assert torch.isfinite(parameter.grad).all(), \
            f"Invalid gradient for {name}"








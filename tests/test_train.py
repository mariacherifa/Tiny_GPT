##############################################################################
#                            Train Test 
##############################################################################

import torch 
import pytest 
from src.train import Train
from src.model import TinyGPT
from src.data import data_split
import math
import numpy as np


# Test 1 : one epoch train

@pytest.mark.parametrize(
    "batch_size, block_size, steps_per_epoch",
    [
        (2, 8, 2),
        (4, 16, 3),
        (8, 32, 5)
    ]
)

def test_train_one_epoch(batch_size, block_size, steps_per_epoch):

    # Configuration 

    n_embd = 32
    vocab_size = 65
    num_heads = 4
    num_layers = 1
    dropout = 0.0
    num_epochs = 2
    val_every = 1
    device = torch.device("cpu")

    text = ( "Hello Brad, thank you for coming, "
        "it is a pleasure to have you here. " )*20

    _, train_data, val_data = data_split(text)

    assert len(train_data) > block_size
    assert len(val_data) > block_size

    model = TinyGPT(n_embd, vocab_size, num_heads, block_size, num_layers, dropout)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    Trainer = Train(model, train_data, val_data, n_embd, batch_size, block_size, 
                    num_epochs,optimizer,steps_per_epoch,val_every,device)

    # Save initial parameters
    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    train_loss_one_epoch = Trainer.train_one_epoch()

    assert math.isfinite( train_loss_one_epoch)

    assert any(
        not torch.equal(w, p)
        for w, p in zip(initial_weights, model.parameters())
    )


# Test 2 : full training function 
@pytest.mark.parametrize(
    "batch_size, block_size, steps_per_epoch",
    [
        (2, 8, 2),
        (4, 16, 3),
        (8, 32, 5)
    ]
)

def test_full_training(batch_size, block_size, steps_per_epoch):

    # Configuration 

    n_embd = 32
    vocab_size = 65
    num_heads = 4
    num_layers = 1
    dropout = 0.0
    num_epochs = 2
    val_every = 1
    device = torch.device("cpu")

    text = ( "Hello Brad, thank you for coming, "
        "it is a pleasure to have you here. "
         "The weather is very cold and rainy in Paris " )*20

    _, train_data, val_data = data_split(text)

    assert len(train_data) > block_size
    assert len(val_data) > block_size

    model = TinyGPT(n_embd, vocab_size, num_heads, block_size, num_layers, dropout)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    Trainer = Train(model, train_data, val_data, n_embd, batch_size, block_size, 
                    num_epochs,optimizer,steps_per_epoch,val_every,device)

    # Save initial parameters
    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    History_training = Trainer.fit()

    assert  len(History_training["Epochs"]) == num_epochs
    assert len(History_training["val_losses"]) == num_epochs
    assert len(History_training["train_losses"]) == num_epochs
    assert np.isfinite(History_training["val_losses"]).all()
    assert np.isfinite(History_training["train_losses"]).all()
    assert any(
        not torch.equal(w, p)
        for w, p in zip(initial_weights, model.parameters())
    )


# Test 3 : test evaluation 

@pytest.mark.parametrize(
    "batch_size, block_size, steps_per_epoch",
    [
        (2, 8, 2),
        (4, 16, 3),
        (8, 32, 5)
    ]
)

def test_evaluation(batch_size, block_size, steps_per_epoch):

    # Configuration 

    n_embd = 32
    vocab_size = 65
    num_heads = 4
    num_layers = 1
    dropout = 0.0
    num_epochs = 2
    val_every = 1
    device = torch.device("cpu")

    text = ( "Hello Brad, thank you for coming, "
        "it is a pleasure to have you here. "
         "The weather is very cold and rainy in Paris " )*20

    _, train_data, val_data = data_split(text)

    assert len(train_data) > block_size
    assert len(val_data) > block_size

    model = TinyGPT(n_embd, vocab_size, num_heads, block_size, num_layers, dropout)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    Trainer = Train(model, train_data, val_data, n_embd, batch_size, block_size, 
                    num_epochs,optimizer,steps_per_epoch,val_every,device)

    # Save initial parameters
    initial_weights = [
        p.detach().clone()
        for p in model.parameters()
    ]

    evaluation_loss = Trainer.evaluate()

    assert math.isfinite(evaluation_loss)
    assert all(
        torch.equal(w, p)
        for w, p in zip(initial_weights, model.parameters())
    )


# Test 4 : test that the model lears (loss decreases)
def test_model_learns():

    torch.manual_seed(42)

    model = TinyGPT(
        n_embd=32,
        vocab_size=65,
        num_heads=4,
        block_size=8,
        num_layers=1,
        dropout=0.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3
    )

    # Fixed batch
    x = torch.randint(0, 65, (2, 8))
    y = torch.randint(0, 65, (2, 8))

    # Initial loss
    model.eval()

    with torch.no_grad():
        initial_loss, _ = model(x, y)

    # Training on the same batch
    model.train()

    for step in range(100):

        optimizer.zero_grad(set_to_none=True)

        loss, _ = model(x, y)

        loss.backward()

        optimizer.step()

    # Final loss
    model.eval()

    with torch.no_grad():
        final_loss, _ = model(x, y)

    assert final_loss.item() < initial_loss.item()


# Test 5: test generated text 
@pytest.mark.parametrize("max_new_tokens", [1, 5, 20])
def test_generate(max_new_tokens):

    torch.manual_seed(42)

    vocab_size = 65

    model = TinyGPT(
        n_embd=32,
        vocab_size=vocab_size,
        num_heads=4,
        block_size=8,
        num_layers=1,
        dropout=0.0
    )

    x = torch.randint(0, vocab_size, (2, 8))

    model.eval()

    with torch.no_grad():
        output = model.generate(x, max_new_tokens)

    # Correct output shape
    assert output.shape == (2, 8 + max_new_tokens)

    # Original tokens unchanged
    assert torch.equal(x, output[:, :8])

    # Valid token IDs
    assert (output >= 0).all()
    assert (output < vocab_size).all()

    # Correct dtype
    assert output.dtype == torch.long
################################################################################
#                             Attention Tests 
################################################################################

import torch
import pytest
from src.data import build_vocab, encode, decode, data_split, get_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test 1 : build vocab test 
@pytest.mark.parametrize("text", ["Hello World"])

def test_vocab(text): 
    vocab, vocab_size, itos, stoi = build_vocab(text)
    encoded_text = encode(text, stoi)
    decoded_text = decode(encoded_text, itos)

    assert sorted(set(text)) == vocab
    assert len(sorted(set(text))) == vocab_size
    assert len(sorted(set(text))) == len(itos)
    assert len(sorted(set(text))) == len(stoi)
    assert decoded_text == text


# Test 2 : Batch shapes 

@pytest.mark.parametrize("text, batch_size, block_size", [ ("Hello World", 2, 2)])

def test_batch(text, batch_size, block_size): 

    data_set, _, _ = data_split(text)

    x,y = get_batch(data_set, batch_size, block_size, device)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)









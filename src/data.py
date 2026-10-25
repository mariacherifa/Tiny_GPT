#########################################################################################
#                                     Data Loading / Split 
#########################################################################################
import torch 

def load_data(path_file, max_chars):
    with open(path_file, "r", encoding="utf-8") as f:
        text = f.read(max_chars)
    return text

def build_vocab(text): 
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    itos = {}
    stoi = {}

    for i, ch in enumerate(vocab): 
        itos[i] = ch 
        stoi[ch] = i 

    return vocab, vocab_size, itos, stoi

def encode(text, stoi): 
    result = []

    for ch in text: 
        result.append(stoi[ch])

    return result 

def decode(ids, itos): 
    result = ""

    for i in ids: 
        result += itos[int(i)]

    return result 

def data_split(text):
    _, _, _, stoi = build_vocab(text)
    data = encode(text, stoi)
    data_set = torch.tensor(data)

    n = len(data_set)

    split = int(0.8*n)

    data_train = data_set[:split]
    data_test = data_set[split:]

    return data_set, data_train, data_test

def get_batch(data, batch_size, block_size, device): 
    if len(data) <= block_size :
        raise ValueError("data is too short for the chosen block_size")

    idx = torch.randint(0,len(data)-block_size,(batch_size,))

    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])

    x = x.to(device)
    y = y.to(device)

    return x,y 







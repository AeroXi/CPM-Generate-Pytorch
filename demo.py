import torch
import numpy as np
from GPT2 import GPT2Model, GPT2Tokenizer

device = 'cuda' #'cuda'

model = GPT2Model(
    vocab_size=30000,
    layer_size=32,
    block_size=1024,
    embedding_dropout=0.0,
    embedding_size=2560,
    num_attention_heads=32,
    attention_dropout=0.0,
    residual_dropout=0.0)

state_dict = torch.load('cpm-1gpu.pth', map_location='cuda')

model.load_state_dict(state_dict)
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer(
    'GPT2/bpe/vocab.json',
    'GPT2/bpe/chinese_vocab.model',
    max_len=512)

def sample(text, max_len=10):
    ids = tokenizer.encode(text)
    input_id = torch.tensor((np.array(ids).reshape(1, -1).astype('int64'))).to(device)
    output, cached_kvs = model(input_id, use_cache=True)
    nid = int(np.argmax(output[0, -1].detach().cpu().numpy()))
    ids += [nid]
    out = [nid]
    for i in range(max_len):
        input_id = torch.tensor(np.array([nid]).reshape(1, -1).astype('int64')).to(device)
        output, cached_kvs = model(input_id, cached_kvs, use_cache=True)
        nid = int(np.argmax(output[0, -1].detach().cpu().numpy()))
        ids += [nid]
        if nid==3:
            break
        out.append(nid)
    print(tokenizer.decode(out))

while True:
    try:
        text = input("输入文本: ")
        sample(text, max_len = 200)
    except:
        break

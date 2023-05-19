import torch

def infer(model, x):

    x_tokens = model.tokenizer(x, return_tensors='pt')
    x_tokens = x_tokens.to('cuda')

    return torch.softmax(
        model(**x_tokens
    ).logits, dim=-1)
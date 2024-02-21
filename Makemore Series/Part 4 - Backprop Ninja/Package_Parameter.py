import torch
from Package_Class import Linear, Tanh, BatchNorm1d

def generate_parameter(vocab_size,settings,g):
  n_embd = 10 
  n_hidden = 100
  block_size = 3

  C = torch.randn((vocab_size, n_embd),generator=g)

  linear_layers = [
    Linear(n_embd * block_size, n_hidden,g, bias=False),
    Linear(n_hidden, n_hidden,g, bias=False),
    Linear(n_hidden, n_hidden,g, bias=False),
    Linear(n_hidden, n_hidden,g, bias=False),
    Linear(n_hidden, n_hidden,g, bias=False),
    Linear(n_hidden, vocab_size,g, bias=False),
  ]

  num_layers = len(linear_layers)
  layers = []
  for i in range(num_layers):
      layer = []
      layer.append(linear_layers[i])

      if i < num_layers - 1 :
        if settings.is_batch_norm_enable:
          layer.append(BatchNorm1d(n_hidden))
        if settings.is_there_activation:
            layer.append(Tanh())
      else:
        if settings.is_batch_norm_enable:
          layer.append(BatchNorm1d(vocab_size))
      layers.extend(layer)

  with torch.no_grad():
    if settings.is_batch_norm_enable:
      layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
      if isinstance(layer, Linear):
        layer.weight *= settings.gain

  parameters = [C] + [p for layer in layers for p in layer.parameters()]
  for p in parameters:
    p.requires_grad = True

  return layers, parameters, C
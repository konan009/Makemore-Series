import matplotlib.pyplot as plt
import torch 
from Package_Class import Linear, Tanh, BatchNorm1d
import torch.nn.functional as F


def plot_graphs(layers,parameters,ud,settings):
    # Create three figures and their respective subplots
    fig1, ax1 = plt.subplots(figsize=(20, 4))
    fig2, ax2 = plt.subplots(figsize=(20, 4))
    fig3, ax3 = plt.subplots(figsize=(20, 4))
    fig4, ax4 = plt.subplots(figsize=(20, 4))

    if settings.is_batch_norm_enable:
        layer_type = Linear
    elif settings.is_there_activation:
        layer_type = Tanh
    else:
        layer_type = Linear

    legends = []
    print(" Statististics for Graph 1")
    for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
        if isinstance(layer, layer_type):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %+f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            ax1.plot(hx[:-1].detach(), hy.detach());
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    ax1.legend(legends);
    ax1.set_title('Distribution');

    print(" ")
    print(" Statististics for Graph 2")
    legends = []
    for i, layer in enumerate(layers[:-1]): 
        if isinstance(layer, layer_type):
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %+f' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            ax2.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__}')
    ax2.legend(legends);
    ax2.set_title('gradient distribution')

    print(" ")
    print(" Statististics for Graph 3")
    legends = []
    for i,p in enumerate(parameters):
        t = p.grad
        if p.ndim == 2:
            ratio = t.std() / p.std()
            print('weight %10s | mean %+f | std %+f | grad:data ratio  %+f' % (tuple(p.shape), t.mean(), t.std(), ratio))
            hy, hx = torch.histogram(t, density=True)
            ax3.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    ax3.legend(legends)
    ax3.set_title('weights gradient ratio');

    legends = []
    for i,p in enumerate(parameters):
        if p.ndim == 2:
            ax4.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
        ax4.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
        ax4.legend(legends);

    # Display the plots
    plt.show()


device = 'cpu'

def generate_data(Xtr,Ytr,layers,C,parameters,settings,g):
    # same optimization as last time
    max_steps = 200000
    batch_size = 32
    lossi = []
    ud = []

    for i in range(max_steps):
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        # forward pass
        emb = C[Xb] # embed the characters into vectors
        x = emb.view(emb.shape[0], -1) # concatenate the vectors
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb) # loss function
        
        # backward pass
        for layer in layers:
            layer.out.retain_grad() # AFTER_DEBUG: would take out retain_graph
            
        for p in parameters:
            p.grad = None
        loss.backward()
        
        # update
        settings.lr =  settings.lr 
        for p in parameters:
            p.data += -settings.lr * p.grad

        # track stats
        if i % 10000 == 0: # print every once in a while
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())
        
        with torch.no_grad():
            ud.append([((settings.lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

        if i >= 1000:
            break # AFTER_DEBUG: would take out obviously to run full optimization
    return ud

    
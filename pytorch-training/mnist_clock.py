import torch
import torch.utils
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torchvision.transforms.functional as FV
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

NUM_EPOCHS = 10
BATCH_SIZE = 32
EMBED_DIM = 128

def embed_number(number: torch.Tensor, d: int, max_pos = 10000, mvPE: int = 2000, device=None):
    max_pos = max_pos * mvPE
    
    if type(number) is not torch.Tensor:
        number = torch.tensor([number], device=device)
    
    device = device if device != None else number.device
    
    ndim = len(number.size())
    if ndim < 2 or number.size(-1) != 1:
        number = number.unsqueeze(-1)
    
    i = torch.arange(start=0, end=d, step=1, dtype=torch.long, device=device)
    k = torch.div(i, 2, rounding_mode='floor')
    
    zero_out_frequency = i * torch.pi * 0.5
        
    # log power rule trick
    base = math.log(max_pos)
    exponent = -(2 * k.float() / d)
    wk = torch.exp(base * exponent)

    radians = wk * number * mvPE
    
    zero_out_odd = torch.round(torch.abs(torch.cos(zero_out_frequency)))
    zero_out_even = torch.round(torch.abs(torch.sin(zero_out_frequency)))
    
    sin = torch.sin(radians) * zero_out_odd
    cos = torch.cos(radians) * zero_out_even
    
    return sin + cos

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 2),
            nn.ReLU(inplace=True),
            nn.Linear(EMBED_DIM * 2, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, rgba: bool = False) -> torch.Tensor:
        x = embed_number(x, EMBED_DIM)
        x = x + torch.randn_like(x)
        x = self.layers.forward(x)
        
        # only for tracing, rgba is not really used as a parameter in ONNX exported model
        if rgba:
            x = (x + 1) * 0.5 * 255
            x = torch.round(x)
            x = x.int()
            x = x.reshape(x.size(0), 28, 28, 1)
            alpha = torch.full_like(x, 255)            
            x = torch.concat([x, x, x, alpha], dim=3)
            x = x.permute(1, 0, 2, 3)
            x = x.contiguous()
            return x
        
        return x.reshape(x.size(0), 1, 28, 28)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FV.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pin_memory = device == 'cuda'

    train_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    mnist = datasets.MNIST('/data', train=True, transform=train_transform, download=True)
    loader = dataloader.DataLoader(
        mnist, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory, 
        pin_memory_device=device, 
        num_workers=torch.get_num_threads(), 
        persistent_workers=True
    )
    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(NUM_EPOCHS):
        for i, (batch, labels) in enumerate(loader):
            batch,labels=batch.to(device),labels.to(device)
            pred = model.forward(labels)
            loss = F.mse_loss(pred, batch)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            if (i + 1) % 100 == 0:
                print(f'[{epoch}] {loss.item()}')
            
    batch = model.forward(torch.concat([torch.arange(10, device=device), torch.arange(10, device=device),torch.arange(10, device=device)]))
    grid = make_grid(batch, padding=0, nrow=5, normalize=True, value_range=(-1,1))
    show(grid)
    plt.show()
    
    dummy_input = (torch.arange(5, dtype=torch.int, device=device), True)
    dynamic_axes = {
        "x": [0],
        "y": [1]
    }
    input_names = [ "x", "rgba" ]
    output_names = [ "y" ]
    torch.onnx.export(model, dummy_input, 'model.onnx', verbose=False, export_params=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
    
if __name__ == '__main__':
    train()
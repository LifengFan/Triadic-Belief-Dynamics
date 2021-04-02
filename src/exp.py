import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Process

torch.multiprocessing.set_start_method('spawn', force="True")

class EDNet(nn.Module):
    def __init__(self):
        super(EDNet, self).__init__()
        self.encoder_1 = nn.Linear(50, 50)
        self.encoder_2 = nn.Linear(50, 50)
        self.decoder = nn.Linear(100, 5)

    def forward(self, x_1, x_2):
        latent_1 = F.dropout(F.relu(self.encoder_1(x_1)), 0.8)
        latent_2 = F.dropout(F.relu(self.encoder_2(x_2)), 0.8)
        x = self.decoder(torch.cat((latent_1, latent_2), 1))
        return x

def torch_fun(net, x):
    return torch.add(x, 1)

def task(net):
    print('Start process')

    x_1 = torch.normal(mean=torch.zeros(5, 50), std=1.0).cuda()
    x_2 = torch.normal(mean=torch.zeros(5, 50), std=1.0).cuda()
    res = net(x_1, x_2)
    print(res.cpu().data)

def main():
    torch.manual_seed(0)
    net = EDNet()
    net = net.cuda()
    net.eval()
    
    pool = []

    for _ in range(5):
        p = Process(target=task, args=(net,))
        p.start()
        pool.append(p)

    for p in pool:
        p.join()

if __name__ == '__main__':
    main()
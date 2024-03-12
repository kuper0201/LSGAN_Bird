import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image

# Set random seed for reproductibility 
manualSeed = 99
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dir_name = "GAN_results"
dataroot = 'bird_data'
workers = 4
batch_size = 30
image_size = 128
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1000
lr = 0.0002
beta1 = 0.5
ngpu = 1

# Create a directory for saving samples
if not os.path.exists(dir_name):
  os.makedirs(dir_name)

dataset = dset.ImageFolder(
  root=dataroot,transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))                            
  ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
  classname = m.__class__.__name__

  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0,0.02)
  
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data,1.0,0.02)
    nn.init.constant_(m.bias.data,0)

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 16),
      nn.ReLU(True),
      # state size. (ngf*16) x 4 x 4
      nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 8 x 8
      nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 16 x 16 
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 32 x 32
      nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf) x 64 x 64
      nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 128 x 128
    )

  def forward(self,input):
    return self.main(input)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      # input is (nc) x 128 x 128
      nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 64 x 64
      nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 32 x 32
      nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 16 x 16 
      nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 8 x 8
      nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(ndf * 16),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*16) x 4 x 4
      nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),

      # DCGAN
      #nn.Sigmoid()

      # LSGAN
      nn.Linear(1, 1, bias=False)

      # state size. 1
    )

  def forward(self, input):
    return self.main(input)

netG = Generator().to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

criterion = nn.MSELoss()

fixed_noise = torch.randn(64,nz,1,1,device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Training...")

#for each epoch
for epoch in range(num_epochs):
  # for each batch
  for i, data in enumerate(dataloader,0):
    # train with all-real batch
    netD.zero_grad()
    real_cpu = data[0].to(device)
    b_size = real_cpu.size(0)
    label = torch.full((b_size,), real_label, device=device)

    output = netD(real_cpu).view(-1)
    errD_real = criterion(output.to(float),label.to(float))
    errD_real.backward()
    D_x = output.mean().item()

    # train with all-fake batch
    noise = torch.randn(b_size,nz,1,1,device=device)
    fake = netG(noise)
    label.fill_(fake_label)

    output = netD(fake.detach()).view(-1)
    errD_fake = criterion(output.to(float),label.to(float))
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    errD = errD_real + errD_fake

    optimizerD.step()

    # Update G
    netG.zero_grad()
    label.fill_(real_label)
    
    output = netD(fake).view(-1)
    errG = criterion(output.to(float),label.to(float))
    errG.backward()
    D_G_z2 = output.mean().item()
    
    optimizerG.step()

    if i % 100 == 0:
      print('[%d/%d][%d/%d] L_D: %.2f L_G: %.2f D(x): %.2f D(G(z)): %.2f / %.2f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
      
      G_losses.append(errG.item())
      D_losses.append(errD.item())
      iters += 1

  torch.save(netD, 'models/discriminator/model_' +str(epoch) +'.pth')
  torch.save(netG, 'models/generator/model_' +str(epoch) +'.pth')

  with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
  img_tmp = vutils.make_grid(fake, padding=2, normalize=True)
  plt.axis("off")
  plt.imshow(np.transpose(img_tmp,(1,2,0)))
  plt.savefig(os.path.join(dir_name, 'GAN_img_{}.png'.format(epoch + 1)))
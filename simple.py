from __future__ import print_function
import argparse
import numpy as np
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
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--testroot', required=True, help='path to testset')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--numClasses', type=int, default=2, help='number of image classes')
parser.add_argument('--numEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
	os.makedirs(opt.outf)
except OSError:
	pass

dataset = dset.ImageFolder(root=opt.dataroot,
							transform=transforms.Compose([
								transforms.Scale(opt.imageSize),
								transforms.CenterCrop(opt.imageSize),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
testset = dset.ImageFolder(root=opt.testroot,
							transform=transforms.Compose([
								transforms.Scale(opt.imageSize),
								transforms.CenterCrop(opt.imageSize),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, 
										shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, 
										shuffle=True, num_workers=1)

class SimpleNet(nn.Module):
	def __init__(self, h_dim, flat_im_size):
		super(SimpleNet, self).__init__()
		self.flat_im_size = flat_im_size
		self.fc1 = nn.Linear(flat_im_size, h_dim)
		self.fc2 = nn.Linear(h_dim, h_dim)
		self.fc3 = nn.Linear(h_dim, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out = self.relu(self.fc1(x.view(-1, self.flat_im_size)))
		out = self.relu(self.fc2(out))
		out = self.sigmoid(self.fc3(out))
		return out.view(-1, 1)

nets = [SimpleNet(400, 3*opt.imageSize**2) for i in range(opt.numClasses)]

input = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
one = torch.FloatTensor([1])
mone = -1*one

if opt.cuda:
	for net in nets:
		net.cuda()
	input, one, mone = input.cuda(), one.cuda(), mone.cuda()

net_optims = [optim.Adam(net.parameters(), lr=opt.lr) for net in nets]

threshold = -1000
for epoch in range(opt.numEpochs):
	for i, data in enumerate(dataloader, 0):
		image, im_class = data
		im_class = im_class[0]
		cur_net = nets[im_class]

		if opt.cuda:
			image = image.cuda()
		input.resize_as_(image).copy_(image)
		inputv = Variable(input)
		
		net_errs = []
		for j, net in enumerate(nets):
			net.zero_grad()
			if j == im_class:
				err = -1*torch.log(cur_net(inputv))
			else:
				err = torch.log(net(inputv))
			if err.data[0][0] < threshold:
				net_errs.append(threshold)
				continue
			err.backward()
			net_errs.append(err.data[0][0])
			net_optims[j].step()

		print('[%d/%d][%d/%d] Loss Cat: %f Loss Dog: %f'
			% (epoch, opt.numEpochs, i, len(dataloader), net_errs[0], net_errs[1]))

corr = 0
for i, data in enumerate(testloader, 0):
	image, im_class = data
	im_class = im_class[0]
	
	if opt.cuda:
		image = image.cuda()
	input.resize_as_(image).copy_(image)
	inputv = Variable(input)

	outputs = [net(inputv).data[0][0] for net in nets]
	if np.argmax(outputs) == im_class:
		corr += 1

print("Accuracy on Cat Dog: ", float(corr)/len(testloader))	

import os
import numpy as np 

import torch
import torch.nn as nn

import skimage.io
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from sklearn.metrics import normalized_mutual_info_score


class MutualInformation(nn.Module):

	def __init__(self, sigma=0.1, num_bins=256, normalize=True):
		super(MutualInformation, self).__init__()

		self.sigma = sigma
		self.num_bins = num_bins
		self.normalize = normalize
		self.eps = 1e-3

		self.bins = nn.Parameter(torch.linspace(0, 255, num_bins).float(), requires_grad=False)

    # inputs values: torch.Size([48, 4096, 3])
    # bins: torch.Size([64])    
	def marginalPdf(self, values):
		# print("inputs values:", values.shape)
		# print("bins:", self.bins.shape)
		residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.eps
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.eps
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''

		# Torch tensors for images between (0, 1)
		input1 = (input1 + 1e-4) * 255
		input2 = (input2 + 1e-4) * 255

		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		if torch.mean(x1) < 1e-3 and torch.mean(x2) < 1e-3:
			return torch.tensor(1.0).to(x1.device)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.eps), dim=1)
		H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.eps), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.eps), dim=(1, 2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = (2 * mutual_information + self.eps) / (H_x1 + H_x2 + self.eps)

		return mutual_information


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W

			return: scalar
		'''
		return self.getMutualInformation(input1, input2)



if __name__ == '__main__':
	
	device = 'cpu'

	### Create test cases ###
	img1 = Image.open('/Users/haochen/Documents/GitHub/FSMNet/decomp/decomp_diffusion/diffusion/grad.jpg').convert('L')
	img2 = img1.rotate(10)

	arr1 = np.array(img1)
	arr2 = np.array(img2)
    
	print("arr1:", arr1.shape)
	print("arr2:", arr2.shape) # inputs values: torch.Size([2, 90000, 1])
    
	mi_true_1 = normalized_mutual_info_score(arr1.ravel(), arr2.ravel())
	mi_true_2 = normalized_mutual_info_score(arr2.ravel(), arr2.ravel())

	img1 = transforms.ToTensor() (img1).unsqueeze(dim=0).to(device)
	img2 = transforms.ToTensor() (img2).unsqueeze(dim=0).to(device)

	# Pair of different images, pair of same images
	input1 = torch.cat([img1, img2])
	input2 = torch.cat([img2, img2])
 
	print("inputs1:", input1.shape, input1.max(), input1.min())  # inputs1: torch.Size([2, 1, 300, 300])

	MI = MutualInformation(num_bins=256, sigma=0.1, normalize=True).to(device)
	mi_test = MI(input1, input2)

	mi_test_1 = mi_test[0].cpu().numpy()
	mi_test_2 = mi_test[1].cpu().numpy()

	print('Image Pair 1 | sklearn MI: {}, this MI: {}'.format(mi_true_1, mi_test_1))
	print('Image Pair 2 | sklearn MI: {}, this MI: {}'.format(mi_true_2, mi_test_2))

	assert(np.abs(mi_test_1 - mi_true_1) < 0.05)
	assert(np.abs(mi_test_2 - mi_true_2) < 0.05)
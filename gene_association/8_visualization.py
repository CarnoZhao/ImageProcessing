import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from collections import OrderedDict
import os

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		# output = output.view(output.size(0), -1)
		output = self.model.avgpool(output)
		output = output.view(output.size(0), -1)
		output = self.model.classifier(output)
		return target_activations, output

def preprocess_image(img):
	# means=[0.485, 0.456, 0.406]
	# stds=[0.229, 0.512, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	# for i in range(3):
	# 	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
	# 	preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	ipt = Variable(preprocessed_img, requires_grad = True)
	return ipt

def draw(real, img, ipt, grad_cam, target_index, savepath):
	cams = []
	for i in range(4):
		mask = grad_cam(ipt, i)
		cams.append(show_cam_on_image(real, img, mask))
	cams = np.concatenate([*cams], axis = 0)
	left = np.ones((512 * 4, 512, 3)) * 255
	left[:512, :, :] = np.uint8(real * 255)
	left[512:1024, :, :] = np.uint8(img * 255)
	cams = np.concatenate([left, cams], axis = 1)
	texts = "a: origin is %s\nb: standardized\nc,g: predict to be huaisi\nd,h: predict to be jizhi\ne,i: predict to be tumor\nf,j: predict to be tumorln" % (os.path.basename(os.path.dirname(savepath)))
	for i, text in enumerate(texts.split('\n')):
		cams = cv2.putText(cams, text, (10, 512 * 2 + 60 + 60 * i), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.2, thickness = 3,color = (0, 0, 0))
	pos = [[0, 1], [0, 2], [1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [2, 3], [2, 4]]
	for i, p in enumerate(pos):
		cams = cv2.putText(cams, chr(ord('a') + i), (p[0] * 512 + 10, p[1] * 512 - 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, thickness = 5, color = (0, 0, 255))
	cv2.imwrite(savepath, cams)

def show_cam_on_image(img, img2, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cam = np.concatenate((cam, heatmap), axis = 1)
	return np.uint8(255 * cam)

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, ipt):
		return self.model(ipt) 

	def __call__(self, ipt, index = None):
		if self.cuda:
			features, output = self.extractor(ipt.cuda())
		else:
			features, output = self.extractor(ipt)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (512, 512))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, ipt):
        positive_mask = (ipt > 0).type_as(ipt)
        output = torch.addcmul(torch.zeros(ipt.size()).type_as(ipt), ipt, positive_mask)
        self.save_for_backward(ipt, output)
        return output

    def backward(self, grad_output):
        ipt, output = self.saved_tensors
        grad_ipt = None

        positive_mask_1 = (ipt > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_ipt = torch.addcmul(torch.zeros(ipt.size()).type_as(ipt), torch.addcmul(torch.zeros(ipt.size()).type_as(ipt), grad_output, positive_mask_1), positive_mask_2)

        return grad_ipt

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model.features._modules.items():
			if module.__class__.__name__ == 'ReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, ipt):
		x = self.model.features(ipt)
		x = self.model.avgpool(x)
		x = x.view(x.size(0), -1)
		return self.model.classifier(x)

	def __call__(self, ipt, index = None):
		if self.cuda:
			output = self.forward(ipt.cuda())
		else:
			output = self.forward(ipt)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = ipt.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/both.png',
	                    help='ipt image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args



if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	# Can work with any model, but it assumes that the model has a 
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	normedimg = os.popen("find /wangshuo/zhaox/ImageProcessing/gene_association/_data/self_normed -name *tif").read().strip().split('\n')
	net = torch.load("/wangshuo/zhaox/ImageProcessing/survival_analysis/_models/FINAL_SURV.model").module
	net.eval()
	mynet = torch.nn.Sequential(OrderedDict([
		("features", torch.nn.Sequential(OrderedDict(list(net.prenet.named_children())[:-2]))),
		("avgpool", net.prenet.avgpool),
		("classifier", net.postnet)
	]))
	mynet.eval()
	root = "/wangshuo/zhaox/ImageProcessing/gene_association/_data/CAM_out/"

	jdic = {"huaisi": 0, "jizhi": 1, "tumor": 2, "tumorln": 3}

	mil_selected = []
	pats = list(set([p.split('/')[-1].split('_')[0] for p in normedimg]))
	finalpreds = []
	for pat in pats:
		imgps = [p for p in normedimg if pat in p]
		preds = []
		for imgp in imgps:
			img = cv2.imread(imgp, 1)
			img = np.float32(cv2.resize(img, (512, 512))) / 255
			ipt = preprocess_image(img)
			pred = net(ipt.cuda()).cpu().data.numpy()[0][0]
			preds.append(pred)
		imgp = imgps[preds.index(max(preds))]
		mil_selected.append(imgp)
		finalpreds.append(max(preds))

	# import pandas as pd
	# df = pd.read_csv("/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/merged.csv")
	spreds = sorted(finalpreds)

	for i in range(len(pats)):
		try:
			pat = pats[i]
			imgp = mil_selected[i]
			# time = df['time'][list(df['number']).index(eval(pat))]
			# event = df['event'][list(df['number']).index(eval(pat))]
			# hosi = df['hosi'][list(df['number']).index(eval(pat))]
			event = -1
			time = -1
			hosi = "NA"
			pred = finalpreds[i]
		except:
			continue
		realp = imgp.replace("self_normed", "sliced")
		prefix = "(%d)%d_%.3f_%.3f_%s.png" %(spreds.index(pred) + 1, event, time, pred, pat)
		savepath = os.path.join(root, prefix)
		os.system("mkdir -p " + os.path.dirname(savepath))
		grad_cam = GradCam(model = mynet, target_layer_names = ["layer4"], use_cuda=True)
		img = cv2.imread(imgp, 1)
		img = np.float32(cv2.resize(img, (512, 512))) / 255
		ipt = preprocess_image(img)
		real = cv2.imread(realp, 1)
		real = np.float32(cv2.resize(real, (512, 512))) / 255
		tp = os.path.basename(imgp).split("_")[1]
		mask = grad_cam(ipt)
		heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		cam = heatmap + np.float32(real)
		cam = cam / np.max(cam)
		cam = np.concatenate((cam, heatmap), axis = 1)
		imgs = np.concatenate((img, real), axis = 1)
		cam = np.concatenate((imgs, cam), axis = 0)
		cam = np.concatenate((cam, np.ones((1024, 512, 3))), axis = 1)
		text = "signature:\n%.6f\ntime:\n%.3f\nevent:\n%d\nhospital:\n%s\npatient number:\n%s\nuse image type:\n%s" %(pred, time, event, hosi, pat, tp)
		for j, t in enumerate(text.split('\n')):
			cam = cv2.putText(cam, t, (1034, 256 + j * 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1.2, color = (255, 0, 0), thickness = 2)
		cv2.imwrite(savepath, np.uint8(cam * 255))

# -*- coding: utf-8 -*-

import glob
import cv2
import numpy as np
import scipy.io
import scipy.misc

images = glob.glob('Images/*/*.jpg')
dogs = {}

for image in images:
	params = image.split('/')
	params = params[1].replace('_', ' ').replace('-', ' ').split(' ')
	cate = ' '.join(params[1:]).lower()
	if not cate in dogs:
		dogs[cate] = []
	dogs[cate].append(image)

width = 12
height = 10
cell = 80
result = np.zeros([height * cell + height - 1, width * cell + width - 1, 3])
i = 0

for key, value in dogs.items():
	image = value[int(np.random.random() * len(value))]
	image_np = cv2.imread(image)
	shape = image_np.shape
	if shape[0] > shape[1]:
		image_np = image_np[shape[0] // 2 - shape[1] // 2: shape[0] // 2 + shape[1] // 2, :, :]
	else:
		image_np = image_np[:, shape[1] // 2 - shape[0] // 2: shape[1] // 2 + shape[0] // 2, :]
	image_np = scipy.misc.imresize(image_np, (cell, cell))
	r = i // width
	c = i % width
	result[r * cell + r : r * cell + cell + r, c * cell + c : c * cell + cell + c, :] = image_np
	i += 1
cv2.imwrite('狗狗图片.png', result)

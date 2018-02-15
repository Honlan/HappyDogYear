# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm

labels = []
for line in tf.gfile.GFile('dog_inception_v3_labels.txt').readlines():
	line = line.strip()
	labels.append(line[10:])

def create_graph():
	graph = tf.Graph()
	graph_def = tf.GraphDef()
	with open('dog_inception_v3_graph.pb', 'rb') as f:
		graph_def.ParseFromString(f.read())
	with graph.as_default():
		tf.import_graph_def(graph_def)
	return graph

def read_images(height=299, width=299, mean=128, std=128):
	images = glob.glob('Images/*/*.jpg')

	file = tf.placeholder(dtype=tf.string)
	file_reader = tf.read_file(file, 'file_reader')
	image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
	image_tensor = tf.cast(image_reader, tf.float32)
	image_tensor = tf.expand_dims(image_tensor, 0)
	image_tensor = tf.image.resize_bilinear(image_tensor, [height, width])
	image_tensor = tf.divide(tf.subtract(image_tensor, [mean]), [std])
	sess = tf.Session()

	datas = []
	trues = []
	for i in tqdm(range(len(images))):
		image = images[i]
		params = image.split('/')
		params = params[1].replace('_', ' ').replace('-', ' ').split(' ')
		datas.append(sess.run(image_tensor, feed_dict={file: image}))
		trues.append(' '.join(params[1:]).lower())

	return datas, trues

fw = open('results.txt', 'w')

graph = create_graph()
with tf.Session(graph=graph) as sess:
	input_operation = sess.graph.get_operation_by_name('import/Mul')
	output_operation = sess.graph.get_operation_by_name('import/final_result')

	datas, trues = read_images()

	for i in tqdm(range(len(datas))):
		data = datas[i]
		prediction = sess.run(output_operation.outputs[0], feed_dict={input_operation.outputs[0]: data})
		prediction = np.squeeze(prediction)
		top_k = prediction.argsort()[-5:][::-1]
		result = ''
		for j in top_k:
			result += '_' + labels[j].lower() + '_' + str(prediction[j])
		fw.write(trues[i] + result + '\n')
fw.close()
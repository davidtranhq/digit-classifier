import struct
import numpy as np

def _read_big_endian_int(s):
	return struct.unpack(">l", s[:4])[0]

def _vector_convert(n):
	# convert n to a 10-dimensional array a with only a[n] = 1.0
	a = np.zeros((10, 1))
	a[n] = 1.0
	return a

def load(image_path, label_path):
	"""Returns a tuple (data) of tuples (image, label), where image is a numpy
	array containing the image data and label is a 10-dimensional numpy
	array containing the expected activation (answer)"""
	with open(image_path, 'rb') as imgf, open (label_path, 'rb') as labelf:
		imgf.seek(4) # num imgs held at byte 4
		labelf.seek(4) # num labels held at byte 4
		num_imgs = _read_big_endian_int(imgf.read(4))
		num_labels = _read_big_endian_int(labelf.read(4))
		# get img size from width stored at byte 8 and height stored at byte 12
		img_size = _read_big_endian_int(imgf.read(4)) * _read_big_endian_int(
			imgf.read(4))
		img_bytes = imgf.read()
		label_bytes = labelf.read()
	# read data into tuple of imgs
	imgs = tuple( np.fromstring(
		img_bytes[i*img_size:(i+1)*img_size], 
		dtype='B').reshape(img_size, 1)/255 for i in range(num_imgs))
	labels = tuple(
		_vector_convert(label_bytes[i]) for i in range(num_labels))
	return [(img, label) for img, label in zip(imgs, labels)]

def load_all():
	training_data = load(
		"samples/train-images.idx3-ubyte", "samples/train-labels.idx1-ubyte")
	test_data = load(
		"samples/t10k-images.idx3-ubyte", "samples/t10k-labels.idx1-ubyte")
	# use 10k training examples as evaluation data
	eval_data = training_data[50000:]
	training_data = training_data[:50000]
	return (training_data, eval_data, test_data)



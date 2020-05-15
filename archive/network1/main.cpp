#include "network.hpp"

// #include <SDL2/SDL.h>

#include <fstream>
#include <iostream>


// reads 4 bits and converts from big-endian to little-endian
uint32_t read_big_endian_int(std::ifstream &f)
{
	uint8_t buf[4] {};
	f.read(reinterpret_cast<char *>(buf), sizeof(uint32_t));
	return (int)buf[0] << 24 | (int)buf[1] << 16 | (int)buf[2] << 8 | (int)buf[3];
}

std::vector<Network::Training_example> load_mnist(
	const std::string& label_path,
	const std::string& img_path)
{
	std::ifstream img_file(img_path, std::ios::binary), 
				  label_file(label_path, std::ios::binary);
	if (!img_file)
		throw std::runtime_error("Unable to open MNIST image file");
	else if (!label_file)
		throw std::runtime_error("Unable to open MNIST label file");
	// file sizes are held at byte 4
	img_file.seekg(4);
	label_file.seekg(4);
	std::vector<Network::Training_example> examples {};
	uint32_t num_labels { 0 }, num_imgs { 0 };
	// get number of images and labels from byte 4 of each file
	num_imgs = read_big_endian_int(img_file);
	num_labels = read_big_endian_int(label_file);
	if (num_imgs != num_labels)
		throw std::runtime_error("Uneven number of labels to images");
	// get image size
	uint32_t img_w { 0 }, img_h { 0 };
	img_w = read_big_endian_int(img_file);
	img_h = read_big_endian_int(img_file);
	size_t img_size { img_w * img_h };
	// read all image, label pairs
	label_file.seekg(8); // data starts at byte 8 in label files
	img_file.seekg(16); // data starts at byte 16 in label files
	for (size_t i = 0; i < num_labels; ++i)
	{
		uint8_t label { 0 };
		std::array<uint8_t, 784> img {};
		label_file.read(reinterpret_cast<char *>(&label), 1);
		img_file.read(reinterpret_cast<char*>(img.data()), 784);
		if (!label_file)
			throw std::runtime_error("Error reading MNIST label file");
		if (!img_file)
			throw std::runtime_error("Error reading MNIST img file");
		examples.push_back(Network::Training_example { label, std::move(img) });
	}
	return examples;
}

int main(int argc, char *argv[])
{
	std::vector<unsigned int> layer_sizes { 784, 30, 10 };
	std::vector<Network::Training_example> training_examples {}, tests {};
	std::string label_path("samples/train-labels.idx1-ubyte"),
		img_path("samples/train-images.idx3-ubyte"),
		test_label_path("samples/t10k-labels.idx1-ubyte"),
		test_img_path("samples/t10k-images.idx3-ubyte");
	training_examples = load_mnist(label_path, img_path);
	std::cout << "Loaded training examples.\n";
	tests = load_mnist(test_label_path, test_img_path);
	std::cout << "Loaded tests.\n";
	Network network(layer_sizes);
	network.train(training_examples, 30, 10, 3.0, tests);
	std::cout << "Finished training.";
	return 0;
}
"""Code to load and test the dataset"""
import random
from read_dataset import MnistDataloader
from pathlib import Path
import matplotlib.pyplot as plt

input_path = Path.cwd().parent / 'data'
training_images_filepath = input_path / 'train-images-idx3-ubyte' /'train-images-idx3-ubyte'
training_labels_filepath = input_path / 'train-labels-idx1-ubyte' /'train-labels-idx1-ubyte'
test_images_filepath = input_path / 't10k-images-idx3-ubyte' /'t10k-images-idx3-ubyte'
test_labels_filepath = input_path / 't10k-labels-idx1-ubyte' /'t10k-labels-idx1-ubyte'

# Helper function
def show_images(images, title_texts):
    cols = 5
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize = (30, 20))
    axes = axes.ravel()
    for idx, (image, title_text) in enumerate(zip(images, title_texts)):
        axes[idx].imshow(image, cmap=plt.cm.gray)
        if title_text:
            axes[idx].set_title(title_text, fontsize=9)
        axes[idx].axis('off')
    for j in range(idx + 1, rows * cols):
        axes[j].axis('off')

    plt.show()

#Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

#Show random training and test images
images_to_show = []
titles_to_show = []

for i in range(0, 10):
    r = random.randint(1, 60000)
    images_to_show.append(x_train[r])
    titles_to_show.append('train image [' + str(r) + '] =' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(1, 10000)
    images_to_show.append(x_test[r])        
    titles_to_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

# Toggle to show images
#show_images(images_to_show, titles_to_show)
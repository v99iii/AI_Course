import matplotlib.pyplot as plt
import keras
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import numpy as np

class Kallback(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.aucs = []
		self.losses = []

	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		y_pred = self.model.predict(self.model.validation_data[0])
		self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return

def plot_history(history):
	plt.figure(figsize=(16,4))

	plt.subplot(1, 2, 1)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Model Loss')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['Training', 'Validation'], loc='upper right')

	plt.subplot(1, 2, 2)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Model Accuracy')
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.legend(['Training', 'Validation'], loc='lower right')

	plt.show()

def plot_samples(samples, predicted):
    no_examples = predicted.shape[0]
    plt.figure(figsize=(16,8))
    for i,s in enumerate(samples):
        plt.subplot(1, no_examples, i+1)
        plt.imshow(samples[i], interpolation='nearest')
        plt.text(0, 0, predicted[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
        plt.axis('off')


def plot_2D_manifold(images, rows):
    nx = rows
    ny = int(np.ceil(images.shape[0]/rows))
    digit_size = images.shape[1]
    figure = np.zeros((digit_size * nx, digit_size * ny))
    # Linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z,
    # since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, nx))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, ny))
    idx = 0

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            if not(idx>=images.shape[0]):
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = images[idx,:,:]
                idx+=1

    #     print(i,j)
    # print(i,j)

    plt.figure(figsize=(40, 40))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

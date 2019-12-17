from models import *
import os


if __name__ == '__main__':

	os.system('rm -rf images')
	os.system('mkdir images')

	latent_dim = 100

	discriminator = discriminator()
	generator = generator(latent_dim)
	gan = gan(generator, discriminator)

	dataset = fetch_data_and_preprocess()

	train(generator, discriminator, gan, dataset, latent_dim, interval=20, save_image=True)

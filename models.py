from utils import *

def discriminator(in_shape=(28,28,1)):
	model = Sequential()

	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))

	opt=Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

def generator(latent_dim):

	model=Sequential()

	n_nodes=128*7*7

	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7,7,128)))

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))
	return model

def gan(generator, discriminator):

	discriminator.trainable=False

	model=Sequential()
	model.add(generator)
	model.add(discriminator)

	opt=Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

def train(gen, disc, gan, dataset, latent_dim, epoch=100, no_of_batch=128, interval=100, save_image=False):

	batch_per_epoch = dataset.shape[0]//no_of_batch
	half_batch = no_of_batch//2
	cnt=1

	for i in range(epoch):
		for j in range(batch_per_epoch):
			x_real, y_real = generate_real_samples(dataset, half_batch)
			# print(x_real.shape,y_real.shape,sep=" ")
			# print(x_real[0])
			d_loss1, _ = disc.train_on_batch(x_real, y_real)
			# #print('Hi')
			x_fake, y_fake = generate_fake_samples(gen, latent_dim, half_batch)
			# print(x_fake.shape,y_fake.shape,sep=" ")
			d_loss2, _ = disc.train_on_batch(x_fake, y_fake)

			x_gan = generate_latent_points(latent_dim, no_of_batch)
			y_gan = np.ones((no_of_batch, 1))

			g_loss = gan.train_on_batch(x_gan, y_gan)

			if(j%interval==0 and save_image):
				generate_image(gen,latent_dim,cnt)
				cnt+=1

			print(str(i+1)+","+str(j+1)+" "+str(batch_per_epoch)+"---------->"+str(d_loss1)+"   "+str(d_loss2)+"   "+str(g_loss))

	g_model.save('generator.h5')


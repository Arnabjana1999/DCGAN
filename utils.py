from imports import *
from PIL import Image 


def fetch_data_and_preprocess():

	(train_x,_),(_,_) = load_data()

	x = np.expand_dims(train_x, axis=-1)

	x=x.astype('float32')

	x=(x-127.5)/127.5    #mean-centre data from [0,255] to [-1,1]
	return x 

def generate_real_samples(dataset, n):

	# print("real : ",dataset.shape)

	ix=np.random.randint(0,dataset.shape[0],n)
	# randomly selected images from dataset
	x = dataset[ix]
	# class labels
	y=np.ones((n,1))
	return x,y

def generate_latent_points(latent_dim, n):

	x_input = np.random.randn(latent_dim*n)
	x_input = x_input.reshape(n, latent_dim)
	return x_input

def generate_fake_samples(generator, latent_dim, n):

	x_input = generate_latent_points(latent_dim,n)
	# print(x_input.shape)
	x=generator.predict(x_input)
	# print("Fake : ",x.shape)
	y=np.zeros((n,1))
	return x,y

def generate_image(generator, latent_dim, i):

	arr,_ = generate_fake_samples(generator,latent_dim,1)
	arr=arr[0]
	arr = arr*127.5+127.5
	arr=arr.astype('int')[:,:,0]
	img = Image.fromarray(arr,mode='L')
	img.save('images/'+str(i)+'.png')


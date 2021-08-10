import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree, cKDTree
from sklearn.cluster import MiniBatchKMeans, KMeans, MeanShift, AffinityPropagation, AgglomerativeClustering
from sklearn import metrics 
from scipy.stats import entropy
#from topk import SmoothTop1SVM
import argparse
import warnings
warnings.filterwarnings("ignore")
 
argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-p', '--pool', help='pooling algorithm',type=str, default='att')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='resnet34')
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=True)
parser.add_argument('-a', '--alpha', help='weight of the WSI loss: ',type=float, default=1.0)
parser.add_argument('-i', '--input_folder', help='path of the folder where train.csv and valid.csv are stored',type=str, default='./partition/')
parser.add_argument('-o', '--output_folder', help='path where to store the model weights',type=str, default='./models/')
parser.add_argument('-w', '--wsi_folder', help='path where WSIs are stored',type=str, default='./images/')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
TASK = args.TASK
MAGNIFICATION = args.MAG
EMBEDDING_bool = args.features
ALPHA = args.alpha
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
WSI_FOLDER = args.wsi_folder

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

free_memory = False

print("PARAMETERS")
print("TASK: " + str(TASK))
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("POOLING ALGORITHM: " + str(pool_algorithm))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

#create folder (used for saving weights)
def create_dir(models_path):
	if not os.path.isdir(models_path):
		try:
			os.mkdir(models_path)
		except OSError:
			print ("Creation of the directory %s failed" % models_path)
		else:
			print ("Successfully created the directory %s " % models_path)

def select_parameters_colour():
	hue_shift_limit=(-9,9)
	sat_shift_limit=(-25,25)
	val_shift_limit=(-10,10)
	
	p1 = np.random.uniform(-9,9,1)
	p2 = np.random.uniform(-25,25,1)
	p3 = np.random.uniform(-10,10,1)
	
	return p1[0],p2[0],p3[0]

def generate_transformer(prob = 0.5):
	list_operations = []
	probas = np.random.rand(4)
	
	if (probas[0]>prob):
		#print("VerticalFlip")
		list_operations.append(A.VerticalFlip(always_apply=True))
	if (probas[1]>prob):
		#print("HorizontalFlip")
		list_operations.append(A.HorizontalFlip(always_apply=True))
	if (probas[2]>prob):
		#print("RandomRotate90")
		list_operations.append(A.RandomRotate90(always_apply=True))
	if (probas[3]>prob):
		#print("HueSaturationValue")
		p1, p2, p3 = select_parameters_colour()
		list_operations.append(A.HueSaturationValue(always_apply=True,hue_shift_limit=(p1,p1+1e-4),sat_shift_limit=(p2,p2+1e-4),val_shift_limit=(p3,p3+1e-4)))
		
	pipeline_transform = A.Compose(list_operations)
	return pipeline_transform


def generate_list_instances(filename):

	instance_dir = WSI_FOLDER
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'
	
	return instance_csv 

#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")
#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

create_dir(OUTPUT_FOLDER)

models_path = OUTPUT_FOLDER
checkpoint_path = models_path+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = models_path+'MIL_colon_'+TASK+'.pt'
model_weights_filename_temporary = models_path+'MIL_colon_'+TASK+'_temporary.pt'

#CSV LOADING
print("CSV LOADING ")
csv_folder = INPUT_FOLDER

if (TASK=='binary'):

	N_CLASSES = 1
	#N_CLASSES = 2

	if (N_CLASSES==1):
		csv_filename_training = csv_folder+'train_binary.csv'
		csv_filename_validation = csv_folder+'valid_binary.csv'

	
elif (TASK=='multilabel'):

	N_CLASSES = 5

	csv_filename_training = csv_folder+'train_multilabel.csv'
	csv_filename_validation = csv_folder+'valid_multilabel.csv'

	
#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values#[:10]
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values#[:10]


class ImbalancedDatasetSampler_multilabel(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
		
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			for l in label:
				if l in label_to_count:
					label_to_count[l] += 1
				else:
					label_to_count[l] = 1
	
		# weight for each sample
		weights = []

		for idx in self.indices:
			c = 0
			for l in self._get_label(dataset, idx):
				c = c+(1/label_to_count[l])
			weights.append(c)
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		labels = np.where(dataset[idx,1:]==1)[0]
		#labels = dataset[idx,2]
		return labels
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

class ImbalancedDatasetSampler_single_label(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

#MODEL DEFINITION
pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', CNN_TO_USE, pretrained=True)
if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

class MIL_model(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(MIL_model, self).__init__()
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = N_CLASSES
		
		if (EMBEDDING_bool==True):

			if ('resnet34' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES

			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)

		else:
			self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)

			if ('resnet34' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.L = self.E
				self.D = 256
				self.K = self.N_CLASSES

		self.attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, self.K)
		)
		
		self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.L*self.K, self.K)
        )
	

	def forward(self, x, conv_layers_out):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		#if used attention pooling
		A = None
		#m = torch.nn.Softmax(dim=1)
		m_binary = torch.nn.Sigmoid()
		m_multiclass = torch.nn.Softmax()
		dropout = torch.nn.Dropout(p=0.2)
		
		if x is not None:
			#print(x.shape)
			conv_layers_out=self.conv_layers(x)
			#print(x.shape)
			
			conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

		if ('mobilenet' in CNN_TO_USE):
			dropout = torch.nn.Dropout(p=0.2)
			conv_layers_out = dropout(conv_layers_out)


		if (EMBEDDING_bool==True):
			embedding_layer = self.embedding(conv_layers_out)
			features_to_return = embedding_layer
		else:
			embedding_layer = conv_layers_out
			features_to_return = embedding_layer

		#print(features_to_return.size())

		A = self.attention(features_to_return)
		A = torch.transpose(A, 1, 0)
		A = F.softmax(A, dim=1)

		#print(A.size())

		M = torch.mm(A, features_to_return)

		#print(M.size())

		if (TASK=='multilabel'):
			M = M.view(-1, self.L*self.K)

		#print(M.size())

		Y_prob = self.classifier(M)

		#print(Y_prob.size())

		#print(output_fcn.shape)

		output_pool = m_binary(Y_prob)
			
		output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)
		#print(output_fcn.size())
		
		return output_pool, A, features_to_return
		#return output_pool, output_fcn, A, features_to_return, instances_attn, labels_attn, instances_cl, labels_cl 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = torch.load(model_weights_pretrained_filename)
model = MIL_model()
model.to(device)

from torchvision import transforms
prob = 0.5
pipeline_transform_cluster = A.Compose([
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	#A.ElasticTransform(alpha=0.1,p=prob),
	A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-30,20),val_shift_limit=(-15,15),p=prob),
	])

preprocess = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, partition, pipeline_transform):
		self.list_IDs = list_IDs
		self.set = partition
		self.pipeline_transform = pipeline_transform

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index][0]
		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)

		if (self.set == 'train'):
			#data augmentation
			X = self.pipeline_transform(image=X)['image']

		#data transformation
		input_tensor = preprocess(X).type(torch.FloatTensor)
				
		#return input_tensor
		return input_tensor
	
class Dataset_bag(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		instances_filename = generate_list_instances(ID)
		y = self.labels[index]
		if (TASK=='binary' and N_CLASSES==1):
			y = np.asarray(y)
		else:
			y = torch.tensor(y.tolist() , dtype=torch.float32)

				
		return instances_filename, y



batch_size_bag = 1

if (TASK=='binary' and N_CLASSES==1):
	sampler = ImbalancedDatasetSampler_single_label
	params_train_bag = {'batch_size': batch_size_bag,
		  'sampler': sampler(train_dataset)}
		  #'shuffle': True}

elif (TASK=='binary' and N_CLASSES==2):
	sampler = ImbalancedDatasetSampler_multilabel
	params_train_bag = {'batch_size': batch_size_bag,
		  'sampler': sampler(train_dataset)}
		  #'shuffle': True}
	
elif (TASK=='multilabel'):
	sampler = ImbalancedDatasetSampler_multilabel
	params_train_bag = {'batch_size': batch_size_bag,
		  'sampler': sampler(train_dataset)}
		  #'shuffle': True}

params_valid_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}

params_test_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}

num_epochs = EPOCHS

if (TASK=='binary' and N_CLASSES==1):
	training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1])
	training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

	validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1])
	validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)


else:
	training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1:])
	training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

	validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1:])
	validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)


# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

if (TASK=='binary' and N_CLASSES==1):
	criterion_wsi = torch.nn.BCELoss()
	criterion_patches = torch.nn.BCELoss()


elif (TASK=='multilabel'):
	class_sample_count = [0,0,0,0,0]

	for i in range(len(train_dataset)):
		class_sample_count = class_sample_count + train_dataset[i,1:]

	class_sample_count = np.array(class_sample_count)
	weight = (class_sample_count[-1] / class_sample_count).astype(np.float)
	samples_weight = torch.from_numpy(weight).type(torch.FloatTensor)
	#criterion_wsi = torch.nn.BCELoss(weight=samples_weight.to(device))
	criterion_wsi = torch.nn.BCELoss()


import torch.optim as optim
optimizer_str = 'adam'
#optimizer_str = 'sgd'

lr_str = '0.01'
lr_str = '0.001'
#lr_str = '0.0001'
#lr_str = '0.00001'

wt_decay_str = '0.0'
#wt_decay_str = '0.1'
#wt_decay_str = '0.05'
#wt_decay_str = '0.01'
wt_decay_str = '0.001'

lr = float(lr_str)
wt_decay = float(wt_decay_str)

if (optimizer_str == 'adam'):
	optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=False)
elif (optimizer_str == 'sgd'):
	optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=wt_decay, nesterov=True)

#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def evaluate_validation_set(model, epoch, generator):
	#accumulator for validation set
	y_pred_val = []
	y_true_val = []

	valid_loss = 0.0

	mode = 'valid'

	tot_batches_valid = len(valid_dataset)

	model.eval()

	with torch.no_grad():
		j = 0
		for inputs_bag,labels in generator:
			print('%d / %d ' % (j, tot_batches_valid))
				#inputs: bags, labels: labels of the bags
			labels_np = labels.cpu().data.numpy()
			len_bag = len(labels_np)

				#list of bags 
			filename_wsi = os.path.split(inputs_bag[0])[1]
			print("inputs_bag " + str(filename_wsi)) 
			inputs_bag = list(inputs_bag)

			for b in range(len_bag):
				labs = []
				labs.append(labels_np[b])
				labs = np.array(labs).flatten()

				labels = torch.tensor(labs).float().to(device)
				labels_wsi_np = labels.cpu().data.numpy()

					#read csv with instances
				csv_instances = pd.read_csv(inputs_bag[b], sep=',', header=None).values
					#number of instances
				n_elems = len(csv_instances)
				print("num_instances " + str(n_elems))
					#params generator instances
				batch_size_instance = BATCH_SIZE

				num_workers = 4
				params_instance = {'batch_size': batch_size_instance,
						'shuffle': True,
						'num_workers': num_workers}

					#generator for instances
				instances = Dataset_instance(csv_instances,'valid',pipeline_transform)
				validation_generator_instance = data.DataLoader(instances, **params_instance)
				
				features = []
				with torch.no_grad():
					for instances in validation_generator_instance:
						instances = instances.to(device)

						# forward + backward + optimize
						feats = model.conv_layers(instances)
						feats = feats.view(-1, fc_input_features)
						feats_np = feats.cpu().data.numpy()
						
						features = np.append(features,feats_np)
						
				#del instances

				features_np = np.reshape(features,(n_elems,fc_input_features))

				if (free_memory):
					del features, feats
				
				inputs = torch.tensor(features_np, requires_grad=True).float().to(device)
			
				predictions, attn_layer, embeddings = model(None, inputs)
				
				try:
					loss = criterion_wsi(predictions, labels)
				except:

					predictions = predictions.view(-1)
					loss = criterion_wsi(predictions, labels)

				outputs_wsi_np = predictions.cpu().data.numpy()

				if (free_memory):
					del attn_layer, embeddings

				valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss))
				
				print('output wsi: '+str(outputs_wsi_np)+', label: '+ str(labels_wsi_np) + ', loss: '+str(valid_loss))

				if(free_memory):
					torch.cuda.empty_cache()

				y_pred_val = np.append(y_pred_val,outputs_wsi_np)
				y_true_val = np.append(y_true_val,labels_np)

				bool_patches = False

			j = j+1
		
	return valid_loss

	#number of epochs without improvement
epoch = 0
if (TASK=='binary'):
	iterations = len(train_dataset)
elif (TASK=='multilabel'):
	iterations = len(train_dataset)#+100

tot_batches_training = iterations#int(len(train_dataset)/batch_size_bag)
best_loss = 100000.0

	#number of epochs without improvement
EARLY_STOP_NUM = 12
early_stop_cont = 0
epoch = 0

validation_checkpoints = checkpoint_path+'validation_losses/'
create_dir(validation_checkpoints)

NUM_WSI_TO_CLUSTER = 1
THRESHOLD = 0.7
#ALPHA = 1

def entropy_uncertaincy(self,prob):
	i = np.argmax(prob)
	v = entropy(prob, base=2)      
	return v

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
	#accumulator loss for the outputs
	train_loss = 0.0

	#accumulator accuracy for the outputs
	acc = 0.0
	mode = 'train'
	#if loss function lower
	is_best = False
	
	model.train()

	dataloader_iterator = iter(training_generator_bag)

	for i in range(iterations):
		print('[%d], %d / %d ' % (epoch, i, tot_batches_training))
		try:
			inputs_bag, labels = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(training_generator_bag)
			inputs_bag,labels = next(dataloader_iterator)
			#inputs: bags, labels: labels of the bags
		labels_np = labels.cpu().data.numpy()
		len_bag = len(labels_np)
		
			#list of bags
		filename_wsi = os.path.split(inputs_bag[0])[1]
		print("inputs_bag " + str(filename_wsi)) 
		inputs_bag = list(inputs_bag)   

			#for each bag inside bags
		for b in range(len_bag):
				#DEFINITION DATA AUGMENTATION (WSI_LEVEL)
			pipeline_transform = generate_transformer()
				#labels
			labs = []
			labs.append(labels_np[b])
			labs = np.array(labs).flatten()

			labels = torch.tensor(labs).float().to(device)
			labels_wsi_np = labels.cpu().data.numpy()
				#instances within the bag
			csv_instances = pd.read_csv(inputs_bag[b], sep=',', header=None).values
				#number of instances

			#filtered_csv = limit_patches(csv_instances,batch_size_instance)
			n_elems = len(csv_instances)
			print("num_instances " + str(n_elems))
			num_workers = 4
			batch_size_instance = BATCH_SIZE
			params_instance = {'batch_size': batch_size_instance,
					'shuffle': True,
					'num_workers': num_workers}
				#generator for instances
			instances = Dataset_instance(csv_instances,'train',pipeline_transform)
			training_generator_instance = data.DataLoader(instances, **params_instance)
					
			#INFERENCE 

			features = []
			model.eval()
			
			with torch.no_grad():
				for instances in training_generator_instance:
					instances = instances.to(device)

					# forward + backward + optimize
					feats = model.conv_layers(instances)
					feats = feats.view(-1, fc_input_features)
					#print(feats.shape)
					feats_np = feats.cpu().data.numpy()
					
					features = np.append(features,feats_np)
					
				#del instances

			features_np = np.reshape(features,(n_elems,fc_input_features))

			if (free_memory):
				torch.cuda.empty_cache()
				del features, feats

			model.train()
			model.zero_grad()
			
			inputs = torch.tensor(features_np, requires_grad=True).float().to(device)
			
			predictions, attn_layer, embeddings = model(None, inputs)
			
			try:
				loss = criterion_wsi(predictions, labels)
			except:

				predictions = predictions.view(-1)
				loss = criterion_wsi(predictions, labels)
									
			loss.backward() 

			outputs_wsi_np = predictions.cpu().data.numpy()
			
			if(free_memory):
				del attn_layer, embeddings

			optimizer.step()
			model.zero_grad()

			train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))
			
			print('output wsi: '+str(outputs_wsi_np)+', label: '+ str(labels_wsi_np) + ', loss: '+str(train_loss))

			#del predictions, labels, inputs
			if(free_memory):
				torch.cuda.empty_cache()

			bool_patches = False
		
		print()
		#i = i+1
	#scheduler.step()

	model.eval()

	print("epoch "+str(epoch)+ " train loss: " + str(train_loss))

	print("evaluating validation")
	valid_loss = evaluate_validation_set(model, epoch, validation_generator_bag)

	#save validation
	filename_val = validation_checkpoints+'validation_value_'+str(epoch)+'.csv'
	array_val = [valid_loss]
	File = {'val':array_val}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

	#save_hyperparameters
	filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
	array_n_classes = [str(N_CLASSES)]
	array_lr = [lr_str]
	array_opt = [optimizer_str]
	array_wt_decay = [wt_decay_str]
	array_embedding = [EMBEDDING_bool]
	array_alpha = [ALPHA]
	File = {'n_classes':array_n_classes,'opt':array_opt, 'lr':array_lr,'wt_decay':array_wt_decay,'embedding':array_embedding,'alpha':array_alpha}

	df = pd.DataFrame(File,columns=['n_classes','opt','lr','wt_decay', 'embedding','alpha'])
	np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')

	if (best_loss>valid_loss):
		early_stop_cont = 0
		print ("=> Saving a new best model")
		print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
		best_loss = valid_loss
		torch.save(model, model_weights_filename)
	else:
		early_stop_cont = early_stop_cont+1
	
	epoch = epoch+1
	if (early_stop_cont == EARLY_STOP_NUM):
		print("EARLY STOPPING")

torch.cuda.empty_cache()
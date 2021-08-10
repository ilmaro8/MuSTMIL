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
import argparse
import warnings
import copy

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
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='resnet34')
parser.add_argument('-l', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-m', '--model', help='path of the model to load',type=str, default='./model/')
parser.add_argument('-i', '--input', help='path of input csv',type=str, default='./model/')
parser.add_argument('-w', '--wsi_folder', help='path where WSIs are stored',type=str, default='./images/')
args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
TASK = args.TASK
MAGNIFICATION = args.MAG
INPUT_DATA = args.input
MODEL_PATH = args.model
WSI_FOLDER = args.wsi_folder

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

EMBEDDING_bool = True

print("PARAMETERS")
print("TASK: " + str(TASK))
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

def generate_list_instances(filename):

	instance_dir = WSI_FOLDER
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'

	return instance_csv 


#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")
checkpoint_path = MODEL_PATH+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = MODEL_PATH

#CSV LOADING

print("CSV LOADING ")
csv_filename_testing = INPUT_DATA
#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

if (TASK=='binary'):
	N_CLASSES = 1
elif (TASK=='multilabel'):
	N_CLASSES = 5


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
		self.N_MAGNIFICATIONS = 2
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

			embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)

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

		attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, self.K)
		)
		
		embedding_fc = torch.nn.Sequential(
            torch.nn.Linear(self.L*self.K, self.K)
        )
		
		attentions = [copy.deepcopy(attention) for i in range(self.N_MAGNIFICATIONS)]
		self.attentions = torch.nn.ModuleList(attentions)

		embeddings = [copy.deepcopy(embedding) for i in range(self.N_MAGNIFICATIONS)]
		self.embeddings = torch.nn.ModuleList(embeddings)

		embeddings_fc = [copy.deepcopy(embedding_fc) for i in range(self.N_MAGNIFICATIONS)]
		self.embeddings_fc = torch.nn.ModuleList(embeddings_fc)


		self.attention_general = torch.nn.Sequential(
				torch.nn.Linear(self.L*self.N_MAGNIFICATIONS, self.D),
				torch.nn.Tanh(),
				torch.nn.Linear(self.D, self.K)
			)

		self.classifier = torch.nn.Linear(in_features=self.L*self.N_MAGNIFICATIONS*self.K, out_features=self.N_CLASSES)

	def forward(self, x, conv_layers_out, mode_eval, idx_scale):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		#if used attention pooling
		A = None
		#m = torch.nn.Softmax(dim=1)
		m_multiclass = torch.nn.Softmax()
		m_binary = torch.nn.Sigmoid()
		dropout = torch.nn.Dropout(p=0.2)

		features = []
		probs = []
		outputs_pool = []
		attentions_values = []		
		
		if x is not None:
			#print(x.shape)
			conv_layers_out=self.conv_layers(x)
			#print(x.shape)
			
			conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

		if ('mobilenet' in CNN_TO_USE):
			dropout = torch.nn.Dropout(p=0.2)
			conv_layers_out = dropout(conv_layers_out)

		if (mode_eval == 'multi_scale'):

			for i in range(self.N_MAGNIFICATIONS):

				attention = self.attentions[i]
				embedding = self.embeddings[i]
				embedding_fc = self.embeddings_fc[i]

				features_layer = conv_layers_out[i].to(device)
				#print(x.shape)

				if ('densenet' in CNN_TO_USE):
					n = torch.nn.AdaptiveAvgPool2d((1,1))
					features_layer = n(features_layer)
				
				if ('mobilenet' in CNN_TO_USE):
					dropout = torch.nn.Dropout(p=0.2)
					features_layer = dropout(features_layer)
				#print(conv_layers_out.shape)

				if (EMBEDDING_bool==True):
					embedding_layer = embedding(features_layer)
					features_to_return = embedding_layer
				else:
					embedding_layer = features_layer
					features_to_return = embedding_layer

				features.append(embedding_layer)
				#print(features_to_return.size())

				A = attention(features_to_return)
				A = torch.transpose(A, 1, 0)
				A = F.softmax(A, dim=1)

				#print(A.size())

				M = torch.mm(A, features_to_return)

				#print(M.size())

				if (TASK=='multilabel'):
					M = M.view(-1, self.L*self.K)

				#print(M.size())

				Y_prob = embedding_fc(M)

				#print(Y_prob.size())

				#print(output_fcn.shape)
				output_pool = m_binary(Y_prob)

				output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)

				attentions_values.append(A)
				outputs_pool.append(output_pool)
				

			all_instances = torch.cat(features, dim=1)

			all_instances = all_instances.to(device)

			features_to_return = all_instances

			#print(features_to_return.size())

			A = self.attention_general(features_to_return)
			A = torch.transpose(A, 1, 0)
			A = F.softmax(A, dim=1)

			#print(A.size())

			M = torch.mm(A, features_to_return)

			#print(M.size())

			if (TASK=='multilabel'):
				M = M.view(-1, self.L*self.N_MAGNIFICATIONS*self.K)

			#print(M.size())

			Y_prob = self.classifier(M)

			#print(Y_prob.size())

			output_pool = m_binary(Y_prob)

			output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)

		elif (mode_eval == 'single_scale'):

			attention = self.attentions[idx_scale]
			embedding = self.embeddings[idx_scale]
			embedding_fc = self.embeddings_fc[idx_scale]

			features_layer = conv_layers_out.to(device)
			#print(x.shape)

			if ('densenet' in CNN_TO_USE):
				n = torch.nn.AdaptiveAvgPool2d((1,1))
				features_layer = n(features_layer)
			
			if ('mobilenet' in CNN_TO_USE):
				dropout = torch.nn.Dropout(p=0.2)
				features_layer = dropout(features_layer)
			#print(conv_layers_out.shape)

			if (EMBEDDING_bool==True):
				embedding_layer = embedding(features_layer)
				features_to_return = embedding_layer
			else:
				embedding_layer = features_layer
				features_to_return = embedding_layer

			#print(features_to_return.size())

			A = attention(features_to_return)
			#print(A.size())
			A = torch.transpose(A, 1, 0)
			A = F.softmax(A, dim=1)

			#print(A.size())

			M = torch.mm(A, features_to_return)

			#print(M.size())

			if (TASK=='multilabel'):
				M = M.view(-1, self.L*self.K)

			#print(M.size())

			Y_prob = embedding_fc(M)

			#print(Y_prob.size())

			output_pool = m_binary(Y_prob)

			output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)

		
		return output_pool, A, features_to_return, outputs_pool, attentions_values, features 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = torch.load(model_weights_pretrained_filename)

from torchvision import transforms
preprocess = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, partition):
		self.list_IDs = list_IDs
		self.set = partition

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index][0]
		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)

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

def accuracy_micro(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    
    return metrics.accuracy_score(y_true_flatten, y_pred_flatten)

    
def accuracy_macro(y_true, y_pred):
    
    n_classes = len(y_true[0])
    
    acc_tot = 0.0
    
    for i in range(n_classes):
        
        acc = metrics.accuracy_score(y_true[:,i], y_pred[:,i])
        #print(acc)
        acc_tot = acc_tot + acc
        
    acc_tot = acc_tot/n_classes
    
    return acc_tot

batch_size_bag = 1

params_test_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}

if (TASK=='binary' and N_CLASSES==1):

	testing_set_bag = Dataset_bag(test_dataset[:,0], test_dataset[:,1])
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_test_bag)
	
	
else:
	
	testing_set_bag = Dataset_bag(test_dataset[:,0], test_dataset[:,1:])
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_test_bag)

mode = 'test'

y_pred = []
y_true = []

filenames_wsis = []
pred_cancers = []
pred_hgd = []
pred_lgd = []
pred_hyper = []
pred_normal = []

if (MAGNIFICATION=='10'):
	m = 0
elif (MAGNIFICATION=='5'):
	m = 1

print(model_weights_filenames_mags[m])

model = torch.load(model_weights_filename)
#model = torch.load(model_weights_filenames_mags[m])
model.to(device)
model.eval()

def save_metric(filename,value):
	array = [value]
	File = {'val':array}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename, df.values, fmt='%s',delimiter=',')

with torch.no_grad():
	j = 0
	for inputs_bag,labels in testing_generator_bag:
			#inputs: bags, labels: labels of the bags
		labels_np = labels.cpu().data.numpy()
		len_bag = len(labels_np)

			#list of bags 
		print("inputs_bag " + str(inputs_bag))
		inputs_bag = list(inputs_bag)

		filename_wsi = inputs_bag[0].split('/')[-2]

		for b in range(len_bag):
			labs = []
			labs.append(labels_np[b])
			labs = np.array(labs).flatten()

			labels = torch.tensor(labs).float().to(device)

				#read csv with instances
			csv_instances = pd.read_csv(inputs_bag[b], sep=',', header=None).values
				#number of instances
			n_elems = len(csv_instances)

				#params generator instances
			batch_size_instance = int(BATCH_SIZE_str)

			num_workers = 4
			params_instance = {'batch_size': batch_size_instance,
					'shuffle': True,
					'num_workers': num_workers}

				#generator for instances
			instances = Dataset_instance(csv_instances,'valid')
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
			
			del features, feats
			
			inputs = torch.tensor(features_np).float().to(device)
			

			predictions, _, _, _, _, _ = model(None, inputs, 'single_scale', m)
			
			outputs_np = predictions.cpu().data.numpy()
			labels_np = labels.cpu().data.numpy()

			#print(outputs_np,labels_np)
			print("["+str(j)+"/"+str(len(test_dataset))+"]")
			print("output: "+str(outputs_np))
			print("ground truth:" + str(labels_np))

			if (TASK=='binary' and N_CLASSES==1):

				filenames_wsis = np.append(filenames_wsis,filename_wsi)
				pred_cancers = np.append(pred_cancers,outputs_np)

			else:

				filenames_wsis = np.append(filenames_wsis,filename_wsi)
				pred_cancers = np.append(pred_cancers,outputs_np[0][0])
				pred_hgd = np.append(pred_hgd,outputs_np[0][1])
				pred_lgd = np.append(pred_lgd,outputs_np[0][2])
				pred_hyper = np.append(pred_hyper,outputs_np[0][3])
				pred_normal = np.append(pred_normal,outputs_np[0][4])

			outputs_np = np.where(outputs_np > 0.5, 1, 0)

			torch.cuda.empty_cache()

			y_pred = np.append(y_pred,outputs_np)
			y_true = np.append(y_true,labels_np)

		j = j+1

#MIXED SCALES

kappa_score_general_filename = checkpoint_path+'kappa_score_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
acc_filename = checkpoint_path+'acc_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
acc_macro_filename = checkpoint_path+'acc_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
acc_micro_filename = checkpoint_path+'acc_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
roc_auc_filename = checkpoint_path+'roc_auc_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
f1_score_macro_filename = checkpoint_path+'f1_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
f1_score_micro_filename = checkpoint_path+'f1_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
hamming_loss_filename = checkpoint_path+'hamming_loss_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
recall_score_macro_filename = checkpoint_path+'recall_score_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
recall_score_micro_filename = checkpoint_path+'recall_score_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
jaccard_score_macro_filename = checkpoint_path+'jaccard_score_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
jaccard_score_micro_filename = checkpoint_path+'jaccard_score_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
roc_auc_score_macro_filename = checkpoint_path+'roc_auc_score_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
roc_auc_score_micro_filename = checkpoint_path+'roc_auc_score_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
precision_score_macro_filename = checkpoint_path+'precision_score_macro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
precision_score_micro_filename = checkpoint_path+'precision_score_micro_MS_SS_'+TASK+'_'+MAGNIFICATION+'_WSI_'+DATASET_TO_SELECT+'.csv'
auc_score_filename = checkpoint_path+'auc_score_MS_SS_'+TASK+'_WSI_'+DATASET_TO_SELECT+'.csv'

if (TASK=='binary' and N_CLASSES==1):

	acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
	print("acc " + str(acc))
	save_metric(acc_filename,acc)

	acc_balanced = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred, sample_weight=None, adjusted=False)
	print("acc_balanced " + str(acc_balanced))
	save_metric(acc_balanced_filename,acc_balanced)

	kappa =  metrics.cohen_kappa_score(y_true,y_pred)
	print("kappa " + str(kappa))
	save_metric(kappa_score_general_filename,kappa)

	roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
	print("roc_auc_score " + str(roc_auc_score))
	save_metric(roc_auc_filename,roc_auc_score)

	confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
	print("confusion_matrix ")
	print(str(confusion_matrix))
	save_metric(confusion_matrix_filename,confusion_matrix)

	try:
		fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
		auc_score = metrics.auc(fpr, tpr)
		print("auc_score : " + str(auc_score))
		save_metric(auc_score_filename,auc_score)
	except:
		pass

	try:
		target_names = ['cancer', 'hgd', 'lgd', 'hyper']
		classification_report = metrics.classification_report(y_true, y_pred, target_names=target_names)
		print("classification_report: ")
		print(classification_report)
	except:
		pass
	
	try:
		f1_score_macro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
		f1_score_micro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("f1_score_macro : " + str(f1_score_macro))
		print("f1_score_micro : " + str(f1_score_micro))
		save_metric(f1_score_macro_filename,f1_score_macro)
		save_metric(f1_score_micro_filename,f1_score_micro)
	except:
		pass
	
	try:
		recall_score_macro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro')
		recall_score_micro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("recall_score_macro : " + str(recall_score_macro))
		print("recall_score_micro : " + str(recall_score_micro))
		save_metric(recall_score_macro_filename,recall_score_macro)
		save_metric(recall_score_micro_filename,recall_score_micro)
	except:
		pass
	
	try:
		precision_score_macro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')
		precision_score_micro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("precision_score_macro : " + str(precision_score_macro))
		print("precision_score_micro : " + str(precision_score_micro))
		save_metric(precision_score_macro_filename,precision_score_macro)
		save_metric(precision_score_micro_filename,precision_score_micro)
	except:
		pass
	
	try:
		roc_auc_score_macro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
		roc_auc_score_micro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='micro')
		print("roc_auc_score_macro : " + str(roc_auc_score_macro))
		print("roc_auc_score_micro : " + str(roc_auc_score_micro)) 
		save_metric(roc_auc_score_macro_filename,roc_auc_score_macro)
		save_metric(roc_auc_score_micro_filename,roc_auc_score_macro)
	except:
		pass 

	filename_training_predictions = checkpoint_path+'WSI_predictions_'+DATASET_TO_SELECT+'_MULTISCALE_ON_SINGLE_SCALE_MAGNIFICATION_'+str(MAGNIFICATION)+'x_'+str(TASK)+'.csv'

	File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers}

	df = pd.DataFrame(File,columns=['filenames','pred_cancers'])
	np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')

else:
	y_pred = np.reshape(y_pred,(j,N_CLASSES))
	y_true = np.reshape(y_true,(j,N_CLASSES))
	
	try:
		accuracy_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
		print("accuracy_score : " + str(accuracy_score))
		save_metric(acc_filename,accuracy_score)
	except:
		pass

	try:
		accuracy_macro_score = accuracy_macro(y_true=y_true, y_pred=y_pred)
		print("accuracy_macro_score : " + str(accuracy_macro_score))
		save_metric(acc_macro_filename,accuracy_macro_score)
	except:
		pass
	
	try:
		accuracy_micro_score = accuracy_micro(y_true=y_true, y_pred=y_pred)
		print("accuracy_micro_score : " + str(accuracy_micro_score))
		save_metric(acc_micro_filename,accuracy_micro_score)
	except:
		pass
	
	try:
		hamming_loss = metrics.hamming_loss(y_true=y_true, y_pred=y_pred, sample_weight=None)
		print("hamming_loss : " + str(hamming_loss))
		save_metric(hamming_loss_filename,hamming_loss)
	except:
		pass
	
	try:
		zero_one_loss = metrics.zero_one_loss(y_true=y_true, y_pred=y_pred)
		print("zero_one_loss : " + str(zero_one_loss))
	except:
		pass
	
	try:
		multilabel_confusion_matrix = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
		print("multilabel_confusion_matrix: ")
		print(multilabel_confusion_matrix)
		save_metric(confusion_matrix_filename,multilabel_confusion_matrix)
	except:
		pass

	try:
		target_names = ['cancer', 'hgd', 'lgd', 'hyper']
		classification_report = metrics.classification_report(y_true, y_pred, target_names=target_names)
		print("classification_report: ")
		print(classification_report)
	except:
		pass
	
	try:
		jaccard_score_macro = metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average='macro')
		jaccard_score_micro = metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("jaccard_score_macro : " + str(jaccard_score_macro))
		print("jaccard_score_micro : " + str(jaccard_score_micro))
		save_metric(jaccard_score_macro_filename,jaccard_score_macro)
		save_metric(jaccard_score_micro_filename,jaccard_score_micro)
	except:
		pass
	
	try:
		f1_score_macro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
		f1_score_micro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("f1_score_macro : " + str(f1_score_macro))
		print("f1_score_micro : " + str(f1_score_micro))
		save_metric(f1_score_macro_filename,f1_score_macro)
		save_metric(f1_score_micro_filename,f1_score_micro)
	except:
		pass
	
	try:
		recall_score_macro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro')
		recall_score_micro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("recall_score_macro : " + str(recall_score_macro))
		print("recall_score_micro : " + str(recall_score_micro))
		save_metric(recall_score_macro_filename,recall_score_macro)
		save_metric(recall_score_micro_filename,recall_score_micro)
	except:
		pass
	
	try:
		precision_score_macro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')
		precision_score_micro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro')
		print("precision_score_macro : " + str(precision_score_macro))
		print("precision_score_micro : " + str(precision_score_micro))
		save_metric(precision_score_macro_filename,precision_score_macro)
		save_metric(precision_score_micro_filename,precision_score_micro)
	except:
		pass
	
	try:
		roc_auc_score_macro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
		roc_auc_score_micro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='micro')
		print("roc_auc_score_macro : " + str(roc_auc_score_macro))
		print("roc_auc_score_micro : " + str(roc_auc_score_micro)) 
		save_metric(roc_auc_score_macro_filename,roc_auc_score_macro)
		save_metric(roc_auc_score_micro_filename,roc_auc_score_macro)
	except:
		pass      

	filename_training_predictions = checkpoint_path+'WSI_predictions_'+DATASET_TO_SELECT+'_MULTISCALE_ON_SINGLE_SCALE_MAGNIFICATION_'+str(MAGNIFICATION)+'x_'+str(TASK)+'.csv'

	File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers, 'pred_hgd':pred_hgd,'pred_lgd':pred_lgd, 'pred_hyper':pred_hyper, 'pred_normal':pred_normal}

	df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
	np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')


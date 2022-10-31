import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  
import matplotlib.pyplot as plt
device_gpu=0
batch_size = 1

'''
img_th_rot = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_th_224-002.npy")
labels_th = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_th_224.npy")

img_th_train, img_th_test, labels_th_train, labels_th_test = train_test_split(img_th_rot, labels_th, test_size=0.6, random_state=42,stratify=labels_th)

img_rgb_rot = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_rgb_224.npy")
labels_rgb = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_rgb_224.npy")

img_rgb_train, img_rgb_test, labels_rgb_train, labels_rgb_test = train_test_split(img_rgb_rot, labels_rgb, test_size=0.1, random_state=42,stratify=labels_rgb)
'''
#---- Digits Dataset path------#
img_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target.npy')
labels_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target_labels.npy')

img_rgb_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_source.npy')
labels_rgb_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_source_labels.npy')
#'''

test_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_test), torch.from_numpy(labels_th_test))    
test_dataloader_target = torch.utils.data.DataLoader(test_th, batch_size=batch_size, shuffle=True)

test_rgb = torch.utils.data.TensorDataset(torch.from_numpy(img_rgb_test), torch.from_numpy(labels_rgb_test))    
test_dataloader_source = torch.utils.data.DataLoader(test_rgb, batch_size=batch_size, shuffle=True)

#---- Alexnet model path------#
model = torch.load("/raid/ai21resch11003/DA_HG/models/mnist_mnistm_model1_epoch_66.pth")

model = model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#x = torch.rand(1,3,224,224).cuda(device_gpu)
i = 0
n_total = 0
n_correct = 0

data_target_iter = iter(test_dataloader_target)
len_dataloader = len(test_dataloader_target)
#print("Length of the dataloadetr:" +str(len_dataloader))

bottleneck = np.empty((len_dataloader, 256)) #256 for digit 128*4*4 for aplha
labels_numpy = np.empty((len_dataloader,1))

while i < len_dataloader:
  data_target = data_target_iter.next()
  t_img, t_label = data_target
  t_img = t_img.type(torch.FloatTensor).cuda(device_gpu)
  t_label = t_label.type(torch.LongTensor).cuda(device_gpu)
  
  #print(t_img.shape)
  batch_size = len(t_label)
  
  model.max3.register_forward_hook(get_activation('max3'))
  class_output, _ = model(input_data=t_img, alpha=1.0)
  
  bottleneck[i] = activation['max3'].view(-1, 256)[0].cpu().numpy() #256 for digit 128*4*4 for alpha
  labels_numpy[i] = t_label.cpu().numpy()
  
  pred = class_output.data.max(1, keepdim=True)[1]
  n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
  n_total += batch_size

  i += 1

accu = n_correct.data.numpy() * 1.0 / n_total
print ('epoch: %d, accuracy of the %s dataset: %f' % (1, 'target', accu))
print(bottleneck.shape)
#print(labels_numpy)

## Source

i = 0
n_total = 0
n_correct = 0

data_source_iter = iter(test_dataloader_source)
len_dataloader_source = len(test_dataloader_source)
#print("Length of the dataloadetr:" +str(len_dataloader))

bottleneck_source = np.empty((len_dataloader_source, 256)) #256 for digit
labels_numpy_source = np.empty((len_dataloader_source,1))

while i < len_dataloader_source:
  data_source = data_source_iter.next()
  t_img, t_label = data_source
  t_img = t_img.type(torch.FloatTensor).cuda(device_gpu)
  t_label = t_label.type(torch.LongTensor).cuda(device_gpu)
  
  batch_size = len(t_label)
  
  model.max3.register_forward_hook(get_activation('max3'))
  class_output, _ = model(input_data=t_img, alpha=1.0)
  
  bottleneck_source[i] = activation['max3'].view(-1, 256)[0].cpu().numpy()
  labels_numpy_source[i] = t_label.cpu().numpy()
  
  pred = class_output.data.max(1, keepdim=True)[1]
  n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
  n_total += batch_size

  i += 1

accu_source = n_correct.data.numpy() * 1.0 / n_total
print ('epoch: %d, accuracy of the %s dataset: %f' % (1, 'source', accu_source))
print(bottleneck_source.shape)

for i in range(labels_numpy_source.shape[0]):
  labels_numpy_source[i] += 10
#print(labels_numpy_source) 

bottleneck_f = np.append(bottleneck,bottleneck_source,axis=0)
labels_numpy_f = np.append(labels_numpy,labels_numpy_source,axis=0)

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(bottleneck_f)

df = pd.DataFrame()
df["y"] = labels_numpy_f[:,0]
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

'''
tsne_ = TSNE(n_components=2, verbose=1, random_state=123)
z_src = tsne_.fit_transform(bottleneck_source)

df_src = pd.DataFrame()
df_src["y"] = labels_numpy_source[:,0]
df_src["comp-1"] = z_src[:,0]
df_src["comp-2"] = z_src[:,1]

df_f = df.append(df_src)
'''
sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 20),
                data=df).set(title="Sign Digits Classification dataset T-SNE projection") 
#plt.scatter(df['comp-1'],df['comp-2'],color='red',label='Target')
#plt.scatter(df_src['comp-1'],df_src['comp-2'],color='blue',label='Source')
plt.legend(loc=(0.985,0))
plt.savefig('/raid/ai21resch11003/DA_HG/tsne_digit_comb_alex.png')

'''
model.max3.register_forward_hook(get_activation('max3'))
output = model(x,alpha=1.0)
print(activation['max3'].shape)
'''
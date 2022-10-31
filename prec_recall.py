from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
batch_size=1
cuda=True
device_gpu=1
image_size = 224
alpha=0
my_net = torch.load("/raid/ai21resch11003/DA_HG/mnist_mnistm_model1_epoch_192_best_digit.pth")
my_net = my_net.eval()
'''
img_rgb_resized = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_rgb_224.npy")
labels_rgb = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_rgb_224.npy")
img_th_rot = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_th_224-002.npy")
labels_th = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_th_224.npy")



img_rgb_train, img_rgb_test, labels_rgb_train, labels_rgb_test = train_test_split(img_rgb_resized, labels_rgb, test_size=0.1, random_state=42,stratify=labels_rgb)

img_th_train, img_th_test, labels_th_train, labels_th_test = train_test_split(img_th_rot, labels_th, test_size=0.6, random_state=42,stratify=labels_th)
'''

img_th_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_target.npy')
img_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target.npy')
labels_th_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_target_labels.npy')
labels_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target_labels.npy')

train_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_train), torch.from_numpy(labels_th_train))
test_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_test), torch.from_numpy(labels_th_test))
train_dataloader_target = torch.utils.data.DataLoader(train_th, batch_size=batch_size, shuffle=True)
test_dataloader_target = torch.utils.data.DataLoader(test_th, batch_size=batch_size, shuffle=True)

if cuda:
    my_net = my_net.cuda(device_gpu)

len_dataloader = len(test_dataloader_target)
data_target_iter = iter(test_dataloader_target)

i = 0
n_total = 0
n_correct = 0
pred_ = []
gt_=[]
while i < len_dataloader:

    # test model using target data
    data_target = data_target_iter.next()
    t_img, t_label = data_target

    batch_size = len(t_label)

    input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
    class_label = torch.LongTensor(batch_size)

    if cuda:
        t_img = t_img.cuda(device_gpu)
        t_label = t_label.cuda(device_gpu)
        input_img = input_img.cuda(device_gpu)
        class_label = class_label.cuda(device_gpu)

    input_img.resize_as_(t_img).copy_(t_img)
    class_label.resize_as_(t_label).copy_(t_label)

    class_output, _ = my_net(input_data=input_img, alpha=alpha)
    pred = class_output.data.max(1, keepdim=True)[1]
    n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
    n_total += batch_size
    pred_.append(pred.cpu().numpy())
    gt_.append(class_label.cpu().numpy())

    i += 1

accu = n_correct.data.numpy() * 1.0 / n_total
print ('accuracy :'+str(accu))
temp = np.squeeze(pred_,axis=-1)
temp_ = np.array(gt_)
print(temp_.shape)
print(temp.shape)

#pred = model(img_th_test,1.0)
precision = precision_score(np.squeeze(temp_,axis=-1), np.squeeze(temp,axis=-1),average='micro')
recall = recall_score(np.squeeze(temp_,axis=-1), np.squeeze(temp,axis=-1),average='micro')
class_rep = classification_report(np.squeeze(temp_,axis=-1), np.squeeze(temp,axis=-1))
print('Precision: ',precision)
print('Recall: ',recall)
print(class_rep)
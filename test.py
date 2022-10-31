import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
#from dataset.data_loader import GetLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch

import torchvision.transforms as T
import torchvision
device_gpu=0




def test_(dataset_name, epoch):
    assert dataset_name in ['source', 'target']

    model_root = os.path.join('/raid/ai21resch11003/DA_HG/', 'models')
   

    cuda = True
    cudnn.benchmark = True
    batch_size = 32
    image_size = 224
    alpha = 0
    # Digits and Alpha
    #'''
    img_rgb_resized = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_rgb_224.npy")
    labels_rgb = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_rgb_224.npy")
    img_th_rot = np.load("/raid/ai21resch11003/DA_HG/comb_data/X_th_224-002.npy")
    labels_th = np.load("/raid/ai21resch11003/DA_HG/comb_data/Y_th_224.npy")


    img_rgb_train, img_rgb_test, labels_rgb_train, labels_rgb_test = train_test_split(img_rgb_resized, labels_rgb, test_size=0.1, random_state=42,stratify=labels_rgb)

    img_th_train, img_th_test, labels_th_train, labels_th_test = train_test_split(img_th_rot, labels_th, test_size=0.6, random_state=42,stratify=labels_th)
    #'''
    
    '''
    img_rgb_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_source.npy')
    img_rgb_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_source.npy')
    labels_rgb_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_source_labels.npy')
    labels_rgb_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_source_labels.npy')

    img_th_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_target.npy')
    img_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target.npy')
    labels_th_train = np.load('/raid/ai21resch11003/DA_HG/dataset_244/train_target_labels.npy')
    labels_th_test = np.load('/raid/ai21resch11003/DA_HG/dataset_244/test_target_labels.npy')
    #'''
  


   # '''
    train = torch.utils.data.TensorDataset(torch.from_numpy(img_rgb_train), torch.from_numpy(labels_rgb_train))
    test = torch.utils.data.TensorDataset(torch.from_numpy(img_rgb_test), torch.from_numpy(labels_rgb_test))
    train_dataloader_source = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader_source = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    train_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_train), torch.from_numpy(labels_th_train))
    test_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_test), torch.from_numpy(labels_th_test))
    train_dataloader_target = torch.utils.data.DataLoader(train_th, batch_size=batch_size, shuffle=True)
    test_dataloader_target = torch.utils.data.DataLoader(test_th, batch_size=batch_size, shuffle=True)

    if dataset_name == 'source':
      dataloader= test_dataloader_source
    elif dataset_name == 'target':
      dataloader = test_dataloader_target

    """ training """

    my_net = torch.load(os.path.join(
        model_root, 'mnist_mnistm_model1_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda(device_gpu)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

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

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    return accu
    

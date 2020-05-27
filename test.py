import torch.nn
import torch
from utils.hsidataset import *
from model import HsiNet
from tqdm import tqdm
import spectral
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scikitplot as skplt
import time

target_names_nobg = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
    , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                     'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                     'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                     'Stone-Steel-Towers']
target_names = ['BackGround', 'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
    , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                'Stone-Steel-Towers']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def output(model, testdataloader):
    nb = len(testdataloader)
    pbar = tqdm(enumerate(testdataloader), total=nb)
    label_list = []
    time_list = []
    for t, patch in pbar:
        patch = patch.to(device)
        time_start = time.time()
        predict_prob = model(patch)
        time_end = time.time()
        time_list.append(time_end-time_start)
        predict_list = list(predict_prob[0])
        predict_label = np.argmax(predict_list)
        label_list.append(predict_label)
    label_list = np.reshape(label_list, (145, 145))
    avg_infer_time = 1000*(sum(time_list)/len(time_list))
    print(f"average inference time cost: {avg_infer_time} ms")
    return label_list


def reports(gt, predict):
    gt = np.reshape(gt, (21025,))
    predict = np.reshape(predict, (21025,))
    classi_report = classification_report(gt, predict, target_names=target_names)
    cf_mat = confusion_matrix(gt, predict)
    return classi_report, cf_mat


if __name__ == "__main__":
    testdatasets = HsiDataset("./data", type='out', oversampling=False, removeZeroLabels=False)
    testdataloader = DataLoader(testdatasets, batch_size=1, shuffle=False)
    model = HsiNet(num_class=17)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_file = "./ckpt/best_model_zero.pt"
    model_dict = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    predict_label = output(model, testdataloader)
    gt_label = loadmat(os.path.join('./data/Indian_pines_gt.mat'))['indian_pines_gt']

    gt_img = spectral.imshow(classes=gt_label, figsize=(5, 5))
    plt.show()
    predict_img = spectral.imshow(classes=predict_label, figsize=(5, 5))
    plt.show()
    classi_report, cf_mat = reports(gt_label, predict_label)
    print(classi_report)

    plt.figure(figsize=(15, 15))
    # plot_confusion_matrix(cf_mat, classes=target_names, title="Confusion matrix")
    # plt.show()
    plt.show()

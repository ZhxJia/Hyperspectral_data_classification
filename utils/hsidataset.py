import torch
import torch.nn as nn
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from torchvision import transforms

import pprint

pp = pprint.PrettyPrinter(indent=10)


def applyPCA(X, num_component=30):
    pca_X = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_component, whiten=True)
    pca_X = pca.fit_transform(pca_X)
    pca_X = np.reshape(pca_X, (X.shape[0], X.shape[1], num_component))
    return pca_X


def padWithZeros(X, margin=2):
    padX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    padX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return padX


def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label, :, :, :].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY


class HsiDataset(Dataset):
    def __init__(self, data_dir, type, window_size=5, oversampling=False, removeZeroLabels=False):
        self.data_dir = data_dir
        self.window_size = window_size
        self.type = type
        self.data = loadmat(os.path.join(data_dir, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        self.gt = loadmat(os.path.join(data_dir, 'Indian_pines_gt.mat'))['indian_pines_gt']
        self.pca_X = applyPCA(self.data, num_component=30)
        self.patches_X = []
        self.patches_y = []

        # creat data patches
        patches_X, patches_y = self._create_patches(removeZeroLabels=removeZeroLabels)
        print(f"dataset patches size:{patches_X.shape}   label size: {patches_y.shape}")

        # split data for train and test  stratify:Allocate according to the proportion of categories in y
        X_train, X_test, y_train, y_test = train_test_split(patches_X, patches_y, test_size=0.15,
                                                            random_state=11, stratify=patches_y)
        print(f"train set size:{y_train.shape}  test set size: {y_test.shape}")

        if oversampling:
            sm = SMOTE(random_state=42)
            # X_train, y_train = sm.fit_resample(X_train, y_train)
            X_train, y_train = oversampleWeakClasses(X_train, y_train)
            print(f"after SMOTE oversample, train set size:{X_train.shape} test set size:{y_train.shape}")

        if type == 'train':
            self.patches_X = X_train
            self.patches_y = y_train
        elif type == 'test':
            self.patches_X = X_test
            self.patches_y = y_test
        elif type == "out":
            self.patches_X = patches_X
        else:
            raise ValueError

    def __len__(self):
        return self.patches_X.shape[0]

    def __getitem__(self, item):
        transform = transforms.ToTensor()
        if self.type == 'out':
            patch_ = self.patches_X[item]
            patch_ = transform(patch_).float()
            return patch_
        else:
            patch_ = self.patches_X[item]
            patch_ = transform(patch_).float()
            label_ = self.patches_y[item]
            return patch_, label_

    def _create_patches(self, removeZeroLabels=False):
        windowSize = self.window_size
        margin = int((self.window_size - 1) / 2)
        zeroPaddedX = padWithZeros(self.pca_X, margin=margin)
        patchesData = np.zeros(
            (self.pca_X.shape[0] * self.pca_X.shape[1], windowSize, windowSize, self.pca_X.shape[2]))
        patchesLabels = np.zeros(self.pca_X.shape[0] * self.pca_X.shape[1])
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = self.gt[r - margin, c - margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels > 0, :, :, :]
            patchesLabels = patchesLabels[patchesLabels > 0]
            patchesLabels -= 1
        return patchesData, patchesLabels


if __name__ == "__main__":
    hsi_traindatas = HsiDataset("../data", type='train', oversampling=False)
    hsi_testdatas = HsiDataset("../data", type='test', oversampling=False)
    train_dataloader = DataLoader(hsi_traindatas, batch_size=4, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(hsi_testdatas, batch_size=1, shuffle=True, num_workers=2)
    for i, (patch, label) in enumerate(train_dataloader):
        print(f"{i}:{patch.shape}")

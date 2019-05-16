#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2019/04/13 19:25:09
@Author  :   Wu
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import copy
import json

# import the files of mine
from logger import log, patient_json_dir, is_aug


def evaluate(model, val_loader, device, num_classes, test=True, aug_num=10):
    """
    evaluate the model\\
    Args:
        model: CNN networks
        val_loader: a Dataloader object with validation data
        device: evaluate on cpu or gpu device
        num_classes: the number of classes
    Return:
        classification accuracy of the model on val dataset
    """
    patient_json = {}
    confusion_matrix = np.zeros((num_classes, num_classes + 2))
    # evaluate the model
    model.eval()
    # context-manager that disabled gradient computation
    with torch.no_grad():
        correct = 0
        total = 0

        for ix, (images, targets, ids) in enumerate(val_loader):
            for id in ids:
                if id not in patient_json:
                    patient_json[id] = {
                        'true': 0,
                        'pred': [],
                        'total': 0,
                        'correct': 0,
                        'acc': 0,
                        'prob': [0 for i in range(num_classes)],
                        'vote': 0,
                        'success': 0
                    }

            # data augmentation for test dataset
            if test and False:
                batch_size = targets.shape[0]
                test_transform = transforms.Compose([
                    transforms.RandomRotation(degrees=[-20, 20]),
                    transforms.CenterCrop(size=384),
                    transforms.ToTensor()
                ])
                for i in range(batch_size):
                    pil_image = Image.fromarray(images[i][0].numpy())
                    tensors = [transforms.Compose([transforms.CenterCrop(size=384), transforms.ToTensor()])(pil_image)]
                    for j in range(aug_num):
                        tensors.append(test_transform(pil_image))
                    aug_images = torch.stack(tensors=tensors, dim=0)  # 将增广的图片和原图叠在一起 (stack)
                    aug_targets = torch.tensor([targets[i] for x in range(aug_num + 1)])  # 10 张增广的图片 + 1 原图

                    aug_images = aug_images.to(device)
                    aug_targets = aug_targets.to(device)
                    outputs = model(aug_images)
                    _, predicted = torch.max(outputs.data, dim=1)
                    correct += (predicted == aug_targets).sum().item()
                    total += aug_targets.size(0)
                    # count the class_correct and class_total for each class
                    y_true = [int(x.cpu().numpy()) for x in aug_targets]
                    y_pred = [int(x.cpu().numpy()) for x in predicted]
                    for j in range(len(y_true)):
                        confusion_matrix[y_true[j], y_pred[j]] += 1
                    for j in range(len(y_true)):
                        patient_json[ids[i]]['true'] = y_true[j]
                        patient_json[ids[i]]['pred'].append(y_pred[j])
                    # print(ix, i, aug_images.shape)

            else:
                # device: cpu or gpu
                images = images.to(device)
                targets = targets.to(device)

                # predict with the model
                outputs = model(images)

                # return the maximum value of each row of the input tensor in the
                # given dimension dim, the second return vale is the index location
                # of each maxium value found(argmax)
                _, predicted = torch.max(outputs.data, dim=1)

                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                # count the class_correct and class_total for each class
                y_true = [int(x.cpu().numpy()) for x in targets]
                y_pred = [int(x.cpu().numpy()) for x in predicted]
                for i in range(len(y_true)):
                    confusion_matrix[y_true[i], y_pred[i]] += 1

                for i in range(len(y_true)):
                    patient_json[ids[i]]['true'] = y_true[i]
                    patient_json[ids[i]]['pred'].append(y_pred[i])

        # calculate accuracy
        accuracy = correct / total
        # accuracy for each class
        for i in range(num_classes):
            confusion_matrix[i, -2] = np.sum(confusion_matrix[i, :num_classes])
            confusion_matrix[i, -1] = confusion_matrix[i, i] / confusion_matrix[i, -2]

        log.logger.info('Accuracy on {} set is {}/{} ({:.4f}%)'.format(
            'test ' if test else 'train', correct, total, 100 * accuracy))

        class_acc = []
        for i in range(num_classes):
            log.logger.info('Confusion Matrix on {} set (class {}): {:5d} {:5d} {:5d}    Acc: {}/{} ({:.4f}%)'.format(
                'test ' if test else 'train', i,
                int(confusion_matrix[i, 0]),
                int(confusion_matrix[i, 1]),
                int(confusion_matrix[i, 2]),
                int(confusion_matrix[i, i]),
                int(confusion_matrix[i, -2]),
                100 * float(confusion_matrix[i, -1])
            ))
            class_acc.append(float(confusion_matrix[i, -1]))

            # process patient_json
        for key, _ in patient_json.items():
            patient_json[key]['total'] = len(patient_json[key]['pred'])
            for i in range(patient_json[key]['total']):
                if patient_json[key]['pred'][i] == patient_json[key]['true']:
                    patient_json[key]['correct'] += 1
                patient_json[key]['prob'][patient_json[key]['pred'][i]] += 1
            patient_json[key]['vote'] = patient_json[key]['prob'].index(max(patient_json[key]['prob']))
            patient_json[key]['prob'] = [patient_json[key]['prob'][i] / patient_json[key]['total'] for i in
                                         range(num_classes)]
            patient_json[key]['acc'] = patient_json[key]['correct'] / patient_json[key]['total']
            patient_json[key]['success'] = 1 if patient_json[key]['vote'] == patient_json[key]['true'] else 0

        patient_correct_num = 0
        for key, _ in patient_json.items():
            if patient_json[key]['success'] == 1:
                patient_correct_num += 1
        log.logger.info('Accuracy by patient: {}/{} ({:.4f}%)'.format(patient_correct_num, len(patient_json),
                                                                      100 * patient_correct_num / len(patient_json)))

        with open(patient_json_dir[1] if test else patient_json_dir[0], 'w') as json_file:
            json_file.write(json.dumps(patient_json))

        return accuracy, confusion_matrix, class_acc


def show_curve_1(y1s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    plt.plot(x, y1, label='train')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("{}-aug.png".format(title) if is_aug else "{}-cut.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_2(y1s, y2s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    plt.plot(x, y1, label='train')  # train
    plt.plot(x, y2, label='test')  # test
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("{}-aug.png".format(title) if is_aug else "{}-cut.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')


def show_curve_3(y1s, y2s, y3s, title):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    y2 = np.array(y2s)
    y3 = np.array(y3s)
    plt.plot(x, y1, label='class0')  # class0
    plt.plot(x, y2, label='class1')  # class1
    plt.plot(x, y3, label='class2')  # class2
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    # plt.show()
    plt.savefig("{}-aug.png".format(title) if is_aug else "{}-cut.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure')
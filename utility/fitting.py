#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fitting.py
@Time    :   2019/04/22 13:22:00
@Author  :   Wu
@Version :   2.0
@Desc    :   The `fit` function can shoose loss_func
'''

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import tensorflow as tf

# import the files of mine
from logger import log, tensorboard_dir
import utility.evaluation


def train(model, train_loader, loss_func, optimizer, device):
    """
    train model using loss_fn and optimizer in an epoch.\\
    model: CNN networks\\
    train_loader: a Dataloader object with training data\\
    loss_func: loss function\\
    device: train on cpu or gpu device
    """
    total_loss = 0
    model.train()

    # train the model using minibatch
    for i, (images, targets, _) in enumerate(train_loader):

        #images = np.expand_dims(images, axis=1)
        #images = torch.from_numpy(images)

        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        loss = loss_func(outputs, targets)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # every 10 iteration, print loss
        if (i + 1) % 10 == 0 or i + 1 == len(train_loader):
            log.logger.info("Step [{}/{}] Train Loss: {}".format(i + 1, len(train_loader), loss.item()))

        # count the number of samples of each class in a batch
        # class_sample_num = {}
        # y_true = [int(x.cpu().numpy()) for x in targets]
        # for i in range(len(y_true)):
        #     if y_true[i] not in class_sample_num:
        #         class_sample_num[y_true[i]] = 1
        #     else:
        #         class_sample_num[y_true[i]] += 1
        # log.logger.info('class_sample_num: {}'.format(class_sample_num))

    return total_loss / len(train_loader)


def fit(model, num_epochs, optimizer, device, train_loader, test_loader, train_loader_eval, num_classes, loss_func,
        lr_decay_period=None, lr_decay_rate=2):
    """
    train and evaluate an classifier num_epochs times.\\
    We use optimizer and cross entropy loss to train the model. \\
    Args:
        model: CNN network
        num_epochs: the number of training epochs
        optimizer: optimize the loss function
        device: the device to train
        train_loader: train data loader for training
        test_loader: test data loader for evaluation
        train_loader_eval: train data loader for evaluation
        num_classes: the number of classes
        loss_func: the loss function, loss_func=nn.CrossEntropyLoss() by default
        lr_decay_period, lr_decay_rate: lr decay every `lr_decay_period` epoches by `lr_decay_rate`
    """

    # loss and optimizer
    # loss_func = nn.CrossEntropyLoss()

    model.to(device)
    loss_func.to(device)

    # log train loss and test accuracy
    losses = []
    train_accs = []
    test_accs = []
    train_class_accs = [[], [], []]
    test_class_accs = [[], [], []]

    ######### tensorboard #########
    writer_loss = tf.summary.FileWriter(tensorboard_dir + '/train_loss/')
    writer_acc = [tf.summary.FileWriter(tensorboard_dir + '/train/'), tf.summary.FileWriter(tensorboard_dir + '/test/')]
    writer_train_class = [tf.summary.FileWriter(tensorboard_dir + '/train_class{}/'.format(i)) for i in range(3)]
    writer_test_class = [tf.summary.FileWriter(tensorboard_dir + '/test_class{}/'.format(i)) for i in range(3)]

    log_var = [tf.Variable(0.0) for i in range(4)]
    tf.summary.scalar('train loss', log_var[0])
    tf.summary.scalar('acc', log_var[1])
    tf.summary.scalar('train class acc', log_var[2])
    tf.summary.scalar('test class acc', log_var[3])

    write_op = tf.summary.merge_all()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    ######### tensorboard #########

    for epoch in range(num_epochs):
        log.logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # train step
        loss = train(model, train_loader, loss_func, optimizer, device)
        losses.append(loss)

        log.logger.info('Average train loss in this epoch: {}'.format(loss))

        # evaluate step
        train_accuracy, train_confusion, train_class_acc = utility.evaluation.evaluate(model, train_loader_eval, device,
                                                                                       num_classes, test=False)
        test_accuracy, test_confusion, test_class_acc = utility.evaluation.evaluate(model, test_loader, device,
                                                                                    num_classes, test=True)

        # lr decay
        if lr_decay_period != None:
            if (epoch + 1) % lr_decay_period == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= lr_decay_rate

        ######### tensorboard #########
        # train loss curve
        summary = session.run(write_op, {log_var[0]: float(loss)})
        writer_loss.add_summary(summary, epoch)
        writer_loss.flush()

        # train and test acc curves
        accs = [train_accuracy, test_accuracy]
        for iw, w in enumerate(writer_acc):
            summary = session.run(write_op, {log_var[1]: accs[iw]})
            w.add_summary(summary, epoch)
            w.flush()

        # train class acc curves
        for iw, w in enumerate(writer_train_class):
            summary = session.run(write_op, {log_var[2]: float(train_confusion[iw, -1])})
            w.add_summary(summary, epoch)
            w.flush()

        # test class acc curves
        for iw, w in enumerate(writer_test_class):
            summary = session.run(write_op, {log_var[3]: float(test_confusion[iw, -1])})
            w.add_summary(summary, epoch)
            w.flush()

        # with SummaryWriter(log_dir=tensorboard_dir, comment='train') as writer:
        #     writer.add_scalar('data/train_accuracy', train_accuracy, epoch)
        #     writer.add_scalar('data/test_accuracy', test_accuracy, epoch)

        ######### tensorboard #########

        train_accs.append(train_accuracy)
        test_accs.append(test_accuracy)
        train_class_accs[0].append(train_class_acc[0])
        train_class_accs[1].append(train_class_acc[1])
        train_class_accs[2].append(train_class_acc[2])
        test_class_accs[0].append(test_class_acc[0])
        test_class_accs[1].append(test_class_acc[1])
        test_class_accs[2].append(test_class_acc[2])

    # pyplot curves
    utility.evaluation.show_curve_1(losses, "train loss")
    utility.evaluation.show_curve_2(train_accs, test_accs, "acc")
    utility.evaluation.show_curve_3(
        train_class_accs[0], train_class_accs[1], train_class_accs[2], "train classes acc")
    utility.evaluation.show_curve_3(
        test_class_accs[0], test_class_accs[1], test_class_accs[2], "test classes acc")
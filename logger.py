#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2019/04/05 02:07:02
@Author  :   Wu
@Version :   1.0
@Desc    :   None
'''

# 关于logging：https://cuiqingcai.com/6080.html

import logging
from logging import handlers
import sys
import datetime
import os

PATH_trained_model = './trained_model/'
PATH_log = './log/'
PATH_patient_result = './patient_result/'
PATH_tblogs = './tblogs/'

# create folders
if not os.path.exists(PATH_trained_model):
    os.makedirs(PATH_trained_model)
if not os.path.exists(PATH_log):
    os.makedirs(PATH_log)
if not os.path.exists(PATH_patient_result):
    os.makedirs(PATH_patient_result)
if not os.path.exists(PATH_tblogs):
    os.makedirs(PATH_tblogs)

now_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '='  # 用于以下文件名的命名

is_aug = True
model_name = '{}Model-spc-cut-aug'.format(now_time) if is_aug else '{}Model-spc-cut'.format(
    now_time)  # to save the model
patient_json_dir = ['{}{}patient-train-spc-cut.json'.format(PATH_patient_result, now_time),
                    '{}{}patient-test-spc-cut.json'.format(PATH_patient_result, now_time)]
if is_aug:
    patient_json_dir = ['{}{}patient-train-spc-cut-aug.json'.format(PATH_patient_result, now_time),
                        '{}{}patient-test-spc-cut-aug.json'.format(PATH_patient_result, now_time)]
tensorboard_dir = '{}{}tblog-spc-cut-aug'.format(PATH_tblogs, now_time) if is_aug else '{}{}tblog-spc-cut'.format(
    PATH_tblogs, now_time)
logger_file_path = '{}{}{}.log'.format(PATH_log, now_time, 'logger-spc-cut-aug' if is_aug else 'logger-spc-cut')


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒、M 分、H 小时、D 天、W 每星期（interval==0时代表星期一）、midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


log = Logger(logger_file_path, level='debug')


class ImProgressBar(object):
    def __init__(self, total_iter, bar_len=50):
        self.total_iter = total_iter
        self.bar_len = bar_len
        self.coef = self.bar_len / 100
        self.foo = ['-', '\\', '|', '/']

    def update(self, i):
        sys.stdout.write('\r')
        progress = int((i + 1) / self.total_iter * 100)
        sys.stdout.write("[%4s/%4s] %3s%% |%s%s| %s" % (
            (i + 1),
            self.total_iter,
            progress,
            int(progress * self.coef) * '>',
            (self.bar_len - int(progress * self.coef)) * ' ',
            self.foo[(i + 1) % len(self.foo)]
        ))
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\n')


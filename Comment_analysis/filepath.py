#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   filepath.py    
@Contact :   h939778128@gmail.com
@License :   No license

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/15 21:11   EvanHong      1.0         None
'''

import os
import re


def get_file_path(root_path, file_type):
    """
    获取该文件夹下文件以root path 为开头, file type 为类型的文件的路径
    :param root_path:
    :param file_type:
    :return:
    """
    dirs = []
    file_names = os.listdir(root_path)
    # 打开文件
    for i in range(len(file_names)):
        dir = re.findall('(.*?).{}'.format(file_type), file_names[i])
        if dir:
            dirs.append(str(root_path + '/' + dir[0] + '.' + file_type))
    # 输出所有文件和文件夹
    return dirs
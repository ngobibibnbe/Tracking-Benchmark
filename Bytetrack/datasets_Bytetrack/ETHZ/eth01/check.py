###################################################################
# File Name: check.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 05 Apr 2019 08:58:24 AM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import os.path as osp

label_texts = os.listdir('labels')

for lt in label_texts:
    ltp = osp.join('labels', lt)
    with open(ltp,'r') as f:
        cnt = 0
        for line in f:
            cnt+=1
        if cnt<1:
            print(ltp)

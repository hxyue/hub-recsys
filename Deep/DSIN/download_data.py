'''
@Description:
@version:
@License: MIT
@Author: Wang Yao
@Date: 2020-03-26 16:55:51
@LastEditors: Wang Yao
@LastEditTime: 2020-03-26 17:42:40
'''
import os
import requests
from tqdm import tqdm
from contextlib import closing
from zipfile import ZipFile

# 语料信息
# https://github.com/candlewill/Dialog_Corpus



with ZipFile("xiaohuangji50w_fenciA.conv.zip", "r") as f:
    for filename in f.namelist():
        f.extract(filename, '.')
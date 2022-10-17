# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets

import utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.distributed as dist

import numpy as np
from PIL import Image, ImageFilter
import time
import csv
import sys
import os
import base64
# import cv2
import math
import os.path as op
import random
import io
from glob import glob
import itertools

csv.field_size_limit(sys.maxsize)

ImageFile.LOAD_TRUNCATED_IMAGES = True

def img_from_base64(imagestring, color=True):
    img_str = base64.b64decode(imagestring)
    try:
        if color:
            r = Image.open(io.BytesIO(img_str)).convert('RGB')
            return r
        else:
            r = Image.open(io.BytesIO(img_str)).convert('L')
            return r
    except:
        return None
    
class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_list(self, idxs, q):
        assert isinstance(idxs, list)
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        for idx in idxs:
            pos = self._lineidx[idx]
            self._fp.seek(pos)
            q.put([s.strip() for s in self._fp.readline().split('\t')])

    def close(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _ensure_lineidx_loaded(self):
    #    if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
    #        generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
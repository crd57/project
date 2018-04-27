# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:   map_size
   Author:      crd
   date:        2018/3/26
-------------------------------------------------
"""


def map_size(input_height, filter_height, padding_top, padding_bottom, stride_height):
    out_height = ((input_height - filter_height + padding_top + padding_bottom) / stride_height) + 1
    return out_height
def padding(out)

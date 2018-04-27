# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     erfen
   Description :
   Author :       crd
   date：          2018/2/11
-------------------------------------------------
   Change Activity:
                   2018/2/11:
-------------------------------------------------
"""


def binary_search(lists, item):
    low = 0
    high = len(lists) - 1
    while low <= high:
        mid = int((low + high) / 2)
        guess = lists[mid]
        if item > guess:
            low = mid + 1
        elif item < guess:
            high = mid - 1
        else:
            return mid + 1
    return None


my_list = [2, 4, 6, 8, 11]
print(binary_search(my_list, 4))

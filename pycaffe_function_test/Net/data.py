# coding:utf-8
# __author__ = 'BianXuesheng'
# __data__ = '2016/04/06_17:07 '

from numpy import *

x = random.randint(0, 100, (100, 2))
y = map(lambda x: int(x[-1] > x[0]), x)


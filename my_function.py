# -- coding: utf-8 --

#import pandas as pd
import numpy as np
import datetime

def sayHello():
    print("hello")

def num2datetime(x):
    if type(x) == str:
        return datetime.datetime.strptime(x, '%Y%m%d')
    else:
        return datetime.datetime.strptime(str(int(x)), '%Y%m%d')
    
def get_age(after, before):
    try:
        if after is None or before is None:
            return -1
        
        return int((after - before) / 10000)
    except Exception as ex:
        print(ex)
        return -1
        
import os
import numpy as np
import linecache
import sys

#file_path='E:/Machine-Learning/TensorFlow/CNN.txt'
file_path=input('input the file path:')
rows=1000
count=0

if not os.path.exists(file_path):
    print('the file is not exist,please check the file path')
    sys.exit()
file=open(file_path,'r')
for index, line in enumerate(file):
    count+=1
    
if rows > count:
    file.close()
    print('rows of file:%d is less than %d ' % (count,rows))
    sys.exit()

file.close()
indexs=np.random.randint(0,count,rows)

def check_row(index):
    line=linecache.getline(file_path,index)
    print('checking line % d' % (index))
    '''check this line'''
    pass

for index in indexs:
    check_row(index)

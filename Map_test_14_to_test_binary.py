import os
import numpy as np

file_all = 'dataset/all.txt'
file_bin = 'dataset/binary_test.txt'

f_bin = open(file_bin, 'r')

list_id_bin = []
for x in f_bin:
    x = x.split()
    id_img = x[0]
    list_id_bin += [id_img]
f_bin.close()

f_all = open(file_all, 'r')
list_data_all = []
for x in f_all:
    list_data_all += [x]
f_all.close()

out_file = 'dataset/binary_test_14.txt'
f_out = open(out_file, "w")

for x in list_id_bin:
    for y in list_data_all:
        if x in y:
            f_out.write(y)

f_out.close()
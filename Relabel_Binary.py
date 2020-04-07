import os
import numpy as np
import random
random.seed(1509)

Folder_Path_List = ['images_001/', 'images_002/', 'images_003/', 'images_004/', 'images_005/']
in_file = 'dataset/all.txt'
out_file = 'dataset/binary_all.txt'
download_file = 'dataset/binary_download.txt'

f_in = open(in_file, "r")
f_out = open(out_file, "w")
f_download = open(download_file, "w")

sample_normal = 0
sample_pneu = 0
sample_normal_download = 0
sample_pneu_download = 0

list_names = []

print("Relabeling to Binary ...")

for x in f_in:  
  try:
    x_split = x.split()
    for idx in range(1, len(x_split)):
      x_split[idx] = int(x_split[idx])

    class_array = np.asarray(x_split[1:])
    sum_class = np.sum(class_array)
    if sum_class > 0:
      out_class = 1
      sample_pneu += 1
    else:
      out_class = 0
      sample_normal += 1
    f_out.write(f"{x_split[0]} {out_class}\n")

    if x_split[0][0:11] in Folder_Path_List:
      f_download.write(f"{x_split[0]} {out_class}\n")
      if sum_class > 0:
        sample_pneu_download += 1
      else:
        out_class = 0
        sample_normal_download += 1
      list_names += [f"{x_split[0]} {out_class}\n"]

  except:
    print(x)

f_in.close()
f_out.close()
f_download.close()

print(f"Total normal images: {sample_normal}")
print(f"Total pneumonia images: {sample_pneu}")
print(f"Total normal images downloaded: {sample_normal_download}")
print(f"Total pneumonia images downloaded: {sample_pneu_download}")

# Random subset
random.shuffle(list_names)

total_download = sample_normal_download + sample_pneu_download
train = int(0.7 * total_download)
validate = int(0.1 * total_download)
test = total_download - train - validate

train_set = list_names[0:train]
validate_set = list_names[train:(train+validate)]
test_set = list_names[(train+validate):total_download]

train_name = 'dataset/binary_train.txt'
validate_name = 'dataset/binary_validate.txt'
test_name = 'dataset/binary_test.txt'
f_train = open(train_name, "w")
f_validate = open(validate_name, "w")
f_test = open(test_name, "w")

for x in train_set:
  f_train.write(x)
for x in validate_set:
  f_validate.write(x)
for x in test_set:
  f_test.write(x)

f_train.close()
f_validate.close()
f_test.close()

print("DONE!")
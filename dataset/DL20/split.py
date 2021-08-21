import os
import random

def path_join(train_path, label, file_list):
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(train_path, label, f))
    
    return path_list

def class_balanced_split(ratio=0.2):
    random.seed(10101)
    train_path = "/data_seoul/shhj1998/DL20/train"
    folder_list = os.listdir(train_path) # folder list [0,1,2,...,19]

    label_path_list = []
    unlabel_path_list = []

    for label_num in folder_list:
        file_path = os.path.join(train_path, label_num) 

        file_list = os.listdir(file_path)
        label_list = random.sample(file_list, 100)
        unlabel_list = list(random.sample(set(file_list) - set(label_list), int(100 * ratio)))

        label_path_list += path_join(train_path, label_num, label_list)
        unlabel_path_list += path_join(train_path, label_num, unlabel_list)

    with open(f"/data_seoul/shhj1998/DL20/{str(ratio)}_label_path_list.txt", "w") as f:
        for item in label_path_list:
            f.write("%s\n"%item)
    
    with open(f"/data_seoul/shhj1998/DL20/{str(ratio)}_unlabel_path_list.txt", "w") as f:
        for item in unlabel_path_list:
            f.write("%s\n"%item)
    
    print("label_path_list_len : ", len(label_path_list))
    print("unlabel_path_list_len : ", len(unlabel_path_list))
    print("label data ratio : ", len(label_path_list) / ((len(label_path_list) + len(unlabel_path_list))))

if __name__ == "__main__":
    class_balanced_split(.5)
    class_balanced_split(1.)
    class_balanced_split(1.5)
    class_balanced_split(2.)

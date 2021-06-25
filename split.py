
import os
import random

def path_join(train_path, label, file_list):
    path_list = []
    for f in file_list:
        path_list.append(os.path.join(train_path, label, f))
    
    return path_list

def class_balanced_split(label_ratio=0.2):
    random.seed(10101)
    train_path = "./dataset/DL20/train"
    folder_list = os.listdir(train_path) # folder list [0,1,2,...,19]

    label_path_list = []
    unlabel_path_list = []

    for label_num in folder_list:
        file_path = os.path.join(train_path, label_num) 

        file_list = os.listdir(file_path)
        label_list = random.sample(file_list, int(len(file_list) * label_ratio))
        unlabel_list = list(set(file_list) - set(label_list))

        label_path_list += path_join(train_path, label_num, label_list)
        unlabel_path_list += path_join(train_path, label_num, unlabel_list)

    with open("./dataset/DL20/"+str(label_ratio)+"_label_path_list.txt", "w") as f:
        for item in label_path_list:
            f.write("%s\n"%item)
    
    with open("./dataset/DL20/"+str(label_ratio)+"_unlabel_path_list.txt", "w") as f:
        for item in unlabel_path_list:
            f.write("%s\n"%item)
    
    print("label_path_list_len : ", len(label_path_list))
    print("unlabel_path_list_len : ", len(unlabel_path_list))
    print("label data ratio : ", len(label_path_list) / (len(label_path_list) + len(unlabel_path_list)))

if __name__ == "__main__":
    class_balanced_split(1.0)
    class_balanced_split(0.5)
    class_balanced_split(0.125)
    class_balanced_split(0.05)
    class_balanced_split(0.02)

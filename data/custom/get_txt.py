import os
import random
import sys
path="images/"
train_txt=open("train.txt", "w")
val_txt=open("val.txt", "w")
test_txt=open("test.txt", "w")
files= os.listdir(path)
files.sort()
print(files)
for f in files:
    if os.path.isdir(path+f):
        images=os.listdir(path+f)
        for image in images:
            r=random.randint(0,9)
            line="data/custom/images/"+f+"/"+image+"\n"
            if r>=2:
                train_txt.write(line)
            elif r==1:
                val_txt.write(line)
            else:
                test_txt.write(line)        
train_txt.close()
val_txt.close()









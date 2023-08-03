import os

data_dir = './D0'
train_dir = './dataset/train'
test_dir = './dataset/test'

count = 0
for var in os.listdir(data_dir):
    for img in os.listdir(os.path.join(data_dir, var)):
        count += 1
print(count)

count_train = 0
for var in os.listdir(train_dir):
    for img in os.listdir(os.path.join(train_dir, var)):
        count_train += 1
print(count_train)

count_test = 0
for var in os.listdir(test_dir):
    for img in os.listdir(os.path.join(test_dir, var)):
        count_test += 1
print(count_test)
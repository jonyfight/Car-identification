train_path = './train_file.txt'
val_path = './val_file.txt'

train_data_lines = open(train_path).readlines()
    # Check if image path exists.
train_data_lines = [w for w in train_data_lines]
for subname in train_data_lines:
    subname = subname.strip().split(' ')
    if len(subname) != 2:
        subname = ''.join(subname[:-1])
    else:
        subname  = subname[0]
    if not(subname.endswith(('.jpg','.png','.jpeg','.JPG', '.PNG', '.JPEG'))):
        print(subname)
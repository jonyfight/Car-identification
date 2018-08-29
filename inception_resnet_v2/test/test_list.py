import os

if __name__ == "__main__":

    file_test = '/data0/data/test'
    f = open("./test_file.txt", "w")
    files = os.listdir(file_test)
    num_img = 0
    for subfile in files:
        num_img = len(os.listdir(os.path.join(file_test, subfile)))
        print("{} images of {}".format(num_img, subfile))
        for file in os.listdir(os.path.join(file_test, subfile)):
            if file.endswith(('.jpg','.png','.jpeg','.JPG','.PNG','.JPEG')):
                   try:
                       f.writelines(os.path.join(file_test, subfile, file) + " " + subfile + '\n')
                   except:
                       continue
    f.close()

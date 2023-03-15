import cv2,glob,os,shutil,random
import numpy as np

def move_2_to_3():
    all_imgs_path = glob.glob(r'./data2/*.jpg')
    for img in all_imgs_path:
        count = 0
        start = 0
        end = 0
        real_end = 0
        for x in img:
            if(x == '_'):
                start = count+1
            elif (not(x>='0' and x<='9')) and (start!=0):
                end = count
                break
            count+=1
        count = 0
        for x in img:
            if x=='.':
                real_end = count
            count+=1
        # print(real_end)
        # print(img[start:real_end])
        if(img[end] == '('):
            file_name = './data3/'+img[start-2:real_end]+'.jpg'
            shutil.move(img,'./data3/')
            print(file_name)

def data_enhance():
    all_imgs_path = glob.glob(r'./data3/*.jpg')
    for img in all_imgs_path:
        count = 0
        start = 0
        end = 0
        for x in img:
            if (x == '_'):
                start = count + 1
            elif (not (x >= '0' and x <= '9')) and (start != 0):
                end = count
                break
            count += 1
        # print(img[start:end])
        name = img[start:end] + "(10)"
        file_name = img[:start] + name + img[end:]
        print(file_name)
        os.rename(img, file_name)
        print(start)
        print(end)
        src = cv2.imread(img)
        height = src.shape[0]
        weight = src.shape[1]
        channels = src.shape[2]

        for i in range(height):
            for j in range(weight):
                num = random.randint(0, 200)
                for k in range(channels):
                    if (num > 175):
                        src[i, j, k] = 0
                    elif (num < 10):
                        src[i, j, k] = 255

        src = cv2.flip(src, 1)
        src = cv2.resize(src, (20, 30))
        cv2.imwrite(img, src)

        # for i in range(height):
        #     for j in range(weight):
        #         num = random.randint(0,200)
        #         for k in range(channels):
        #             if(num > 180):
        #                 src[i,j,k] = 0
        #             elif(num < 10):
        #                 src[i,j,k] = 255
        # print(img)
        # save_name = img[:6]+"3"+img[7:]
        # print(save_name)
        # cv2.imwrite(save_name,src)

def change_path_name():
    index = 0
    # original path
    all_imgs_path = glob.glob(r'./data1/5/*.jpg')
    print(all_imgs_path)
    for img in all_imgs_path:
        # new path
        new_name = img[:10] + "5_" + str(index) + ".jpg"
        print(new_name)
        index+=1
        os.rename(img,new_name)

def impulse_noise_7():
    index = 2
    all_imgs_path = glob.glob(r'./data1/7*.png')
    for img in all_imgs_path:
        new_name = img[:9] + "_" + str(index) + img[-4:]
        src = cv2.imread(img)
        src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
        _, src = cv2.threshold(src,0,255,cv2.THRESH_BINARY or cv2.THRESH_OTSU)
        height = src.shape[0]
        weight = src.shape[1]

        for i in range(height):
            for j in range(weight):
                num = random.randint(0, 200)
                src = cv2.flip(src, 1) if num > 100 else src
                if (num > 195):
                    src[i, j] = 0
                elif (num < 5):
                    src[i, j] = 255
        while (new_name in all_imgs_path):
            index += 1
            new_name = img[:9] + "_" + str(index) + img[-4:]
        print(new_name)
        cv2.imwrite(new_name,src)
        index+=1



def impulse_noise_8():
    index = 2
    all_imgs_path = glob.glob(r'./data1/8*.png')
    for img in all_imgs_path:
        new_name = img[:9] + "_" + str(index) + img[-4:]
        src = cv2.imread(img)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, src = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
        height = src.shape[0]
        weight = src.shape[1]

        for i in range(height):
            for j in range(weight):
                num = random.randint(0, 200)
                src = cv2.flip(src, 1) if num > 100 else src
                if (num > 196):
                    src[i, j] = 0
                elif (num < 4):
                    src[i, j] = 255
        while (new_name in all_imgs_path):
            index += 1
            new_name = img[:9] + "_" + str(index) + img[-4:]
        print(new_name)
        cv2.imwrite(new_name, src)
        index+=1

def resize_7():
    data_sizes = [(20, 30), (30, 45), (40, 60), (50, 75), (60, 90), (80, 120)]
    index = 0
    all_imgs_path = glob.glob(r'./data1/7*.png')
    for img in all_imgs_path:
        save_name = img[:9] + "_" + str(index) + img[-4:]
        num = random.randint(0, 5)
        src = cv2.imread(img)
        src = cv2.resize(src, data_sizes[num])
        while (save_name in all_imgs_path):
            index += 1
            save_name = img[:9] + "_" + str(index) + img[-4:]
        print(save_name)
        cv2.imwrite(save_name, src)
        index+=1




def resize_8():
    data_sizes = [(20,30),(30,45),(40,60),(50,75),(60,90),(80,120)]
    index = 0
    all_imgs_path = glob.glob(r'./data1/8*.png')
    for img in all_imgs_path:
        save_name = img[:9] + "_" + str(index) + img[-4:]
        num = random.randint(0,5)  # numpy的randint不包含最大值，python自己的randint包含最大值
        src = cv2.imread(img)
        src = cv2.resize(src, data_sizes[num])
        while(save_name in all_imgs_path):
            index+=1
            save_name = img[:9] + "_" + str(index) + img[-4:]
        print(save_name)
        cv2.imwrite(save_name,src)
        index+=1

if __name__ == "__main__":
    impulse_noise_7()
    impulse_noise_8()
    resize_8()
    resize_7()
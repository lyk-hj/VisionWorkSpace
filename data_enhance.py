import cv2,glob,os,shutil,random
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

all_imgs_path = glob.glob(r'./data3/*.jpg')
for img in all_imgs_path:
    count = 0
    start = 0
    end = 0
    for x in img:
        if(x == '_'):
            start = count+1
        elif (not(x>='0' and x<='9')) and (start!=0):
            end = count
            break
        count+=1
    # print(img[start:end])
    name = img[start:end] + "(10)"
    file_name = img[:start] + name + img[end:]
    print(file_name)
    os.rename(img,file_name)
    print(start)
    print(end)
    src = cv2.imread(img)
    height = src.shape[0]
    weight = src.shape[1]
    channels = src.shape[2]

    for i in range(height):
        for j in range(weight):
            num = random.randint(0,200)
            for k in range(channels):
                if(num > 175):
                    src[i,j,k] = 0
                elif(num < 10):
                    src[i,j,k] = 255

    src = cv2.flip(src,1)
    src = cv2.resize(src,(20,30))
    cv2.imwrite(img,src)


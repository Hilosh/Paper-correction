import cv2
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm

from sklearn import cluster


def ReloadFourPoint(vec):
    #0，1与2，3比较y轴向大小
    if vec[0][1]+vec[1][1]>vec[2][1]+vec[3][1]:#上下点组交换
        vec[0][0], vec[2][0] = vec[2][0], vec[0][0]
        vec[0][1], vec[2][1] = vec[2][1], vec[0][1]
        vec[1][0], vec[3][0] = vec[3][0], vec[1][0]
        vec[1][1], vec[3][1] = vec[3][1], vec[1][1]
    #0，2与1，3比较x轴向大小
    if vec[0][0]+vec[2][0]>vec[1][0]+vec[3][0]:#0，1点交换
        vec[0][0], vec[1][0] = vec[1][0], vec[0][0]
        vec[0][1], vec[1][1] = vec[1][1], vec[0][1]
    else:#2，3点交换
        vec[2][0], vec[3][0] = vec[3][0], vec[2][0]
        vec[2][1], vec[3][1] = vec[3][1], vec[2][1]



def FindFourLines(lines):#待更新
    types = []
    count,_,__ = lines.shape
    for i in range(count):
        types.append([lines[i][0][0],lines[i][0][1]])
    k_means = cluster.KMeans(n_clusters=4)
    k_means.fit(types)
    center = k_means.cluster_centers_
    labels = k_means.labels_
    error = k_means.inertia_

    result = [[0, 0, 0],[0, 0, 0],[0, 0, 0],[0, 0, 0]]
    for i in range(count):
        index = labels[i]
        distance = (lines[i][0][0]-center[index][0])*(lines[i][0][0]-center[index][0])+(lines[i][0][1]-center[index][1])*(lines[i][0][1]-center[index][1])*40000
        if result[index][0]==0 and result[index][1]==0:
            result[index][0]=lines[i][0][0]
            result[index][1]=lines[i][0][1]
            result[index][2] = distance
        if distance<result[index][2]:
            result[index][0] = lines[i][0][0]
            result[index][1] = lines[i][0][1]
            result[index][2] = distance
    for i in range(3):
        for j in range(3-i):
            if result[j][1]>result[j+1][1]:
                result[j][0], result[j + 1][0] = result[j + 1][0], result[j][0]
                result[j][1], result[j + 1][1] = result[j + 1][1], result[j][1]
                result[j][2], result[j + 1][2] = result[j + 1][2], result[j][2]

    # plot points using different colors
    #colors = cm.spectral(labels.astype(float) / n_clusters)
    x1 = [x[0] for x in types]
    y1 = [x[1] for x in types]
    plt.scatter(x1, y1)

    x2 = [x[0] for x in center]
    y2 = [x[1] for x in center]
    plt.scatter(x2, y2, c="red")

    plt.show()
    return result
    # xlim(0, 20)  # limited the length of axis
    #
    # ylim(0, 20)



def DoubleLines(lines):#四条直线已经按角度theta从小到大排序
    Anglediff = 5 #设置一个角度差变量，初始化最大
    goallineIndex = 0 #目标两条直线中的前一条
    for i in range(4):
        tep = min(abs(lines[(i+1)%4][1] - lines[i%4][1]),np.pi+lines[i%4][1]-lines[(i+1)%4][1])
        if tep<Anglediff:
            goallineIndex = i
            Anglediff = tep
            print(Anglediff)
    # 挑选x轴向一组置前
    if abs(lines[goallineIndex % 4][1] - np.pi / 2) < abs(lines[(goallineIndex + 2) % 4][1] - np.pi / 2):
        return ([[lines[goallineIndex % 4]], [lines[(goallineIndex + 1) % 4]]],
                [[lines[(goallineIndex + 2) % 4]], [lines[(goallineIndex + 3) % 4]]])
    else:
        return ([[lines[(goallineIndex + 2) % 4]], [lines[(goallineIndex + 3) % 4]]],
                [[lines[goallineIndex % 4]], [lines[(goallineIndex + 1) % 4]]])



def sortarray(lines):
    count,_,__ = lines.shape
    for i in range(count-1):
        for j in range(count-1-i):
            if lines[j][0][1]>lines[j+1][0][1]:
                lines[j][0][0], lines[j + 1][0][0] = lines[j + 1][0][0], lines[j][0][0]
                lines[j][0][1], lines[j + 1][0][1] = lines[j + 1][0][1], lines[j][0][1]

def imagecut():
    #image_old = cv2.imread("WechatIMG664.jpeg")#高光问题
    #image_old = cv2.imread("IMG_20181203_133538.jpg")#聚类算法问题
    #image_old = cv2.imread("17623C91EA286340F53F74327C7735A0.jpg")#背景色问题
    image_old = cv2.imread("A6671365F7233F0D62003C41F69EDB8D.jpg")

    image = image_old.copy()

    image_old = cv2.cvtColor(image_old, cv2.COLOR_BGR2RGB)

    plt.imshow(image_old)
    plt.show()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (504, 672))

    im_height,im_width,__  = image.shape

    height,width,__ = image_old.shape

    rate_x = width/im_width

    rate_y = height/im_height

    image = cv2.GaussianBlur(image, (9, 9), 0)
    # 使用grabCUT得到前景图片大致位置
    # 设定矩形区域  作为ROI         矩形区域外作为背景
    rect = (20, 20, 480, 650)

    # img.shape[:2]得到img的row 和 col ,
    # 得到和img尺寸一样的掩模即mask ,然后用0填充
    mask = np.zeros(image.shape[:2], np.uint8)

    # 创建以0填充的前景和背景模型,  输入必须是单通道的浮点型图像, 1行, 13x5 = 65的列 即(1,65)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # 调用grabcut函数进行分割,输入图像img, mask,  mode为 cv2.GC_INIT_WITH-RECT
    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    ##调用grabcut得到rect[0,1,2,3],将0,2合并为0,   1,3合并为1  存放于mask2中
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    # 得到输出图像
    out = image * mask2[:, :, np.newaxis]
    plt.imshow(out)
    plt.show()
    out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)


    ret, binary = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 大律法,全局自适应阈值 参数0可改为任意数字但不起作用
    print("阈值：%s" % ret)

    plt.imshow(binary)
    plt.show()
    kernel = np.ones((5, 5), np.uint8)


    #腐蚀膨胀去内外部噪点，再取直线
    binary = cv2.dilate(binary, kernel,iterations=2)#膨胀去除内部噪声
    binary = cv2.erode(binary, kernel, iterations=6)  # 过度腐蚀去除外部噪声
    binary = cv2.dilate(binary, kernel, iterations=12)  # 膨胀弥补损失主体图像
    binary = cv2.erode(binary, kernel, iterations=8)  # 腐蚀
    plt.imshow(binary)
    plt.show()





    #canny算子
    canny = cv2.Canny(binary, 50, 200, apertureSize=3)

    # minLineLength = 100
    # maxLineGap = 10


    #取直线
    lines = cv2.HoughLines(canny, 2, np.pi / 180, 100)
    sortarray(lines)
    print(lines)
    result = image.copy()
    vec = []
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            # 把直线显示在图片上
            # for vector in vec:
            #     if (y2-y1)/(x2-x1)-(vector[1][1]-vector[0][1])/(vector[1][0]-vector[0][0])>0.1:
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0),2)

    plt.imshow(result)
    plt.show()




    reslines = FindFourLines(lines)
    print("reslines:")
    print(reslines)
    doublelines = DoubleLines(reslines)



    result2 = image.copy()

    #取对边直线，各自标色
    lines1,lines2 = doublelines
    print(lines1)
    print(lines2)
    for i in range(2):
        a = np.cos(lines1[i][0][1])
        b = np.sin(lines1[i][0][1])
        x0 = a * lines1[i][0][0]
        y0 = b * lines1[i][0][0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # 把直线显示在图片上
        # for vector in vec:
        #     if (y2-y1)/(x2-x1)-(vector[1][1]-vector[0][1])/(vector[1][0]-vector[0][0])>0.1:
        cv2.line(result2, (x1, y1), (x2, y2), (255, 0, 0), 2)


    for i in range(2):
        a = np.cos(lines2[i][0][1])
        b = np.sin(lines2[i][0][1])
        x0 = a * lines2[i][0][0]
        y0 = b * lines2[i][0][0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # 把直线显示在图片上
        # for vector in vec:
        #     if (y2-y1)/(x2-x1)-(vector[1][1]-vector[0][1])/(vector[1][0]-vector[0][0])>0.1:
        cv2.line(result2, (x1, y1), (x2, y2), (0, 0, 255), 2)





    #取四条直线交叉点
    for i in range(2):
        for j in range(2):
            a = np.cos(lines1[i][0][1])
            b = np.sin(lines1[i][0][1])
            r = lines1[i][0][0]

            a1 = np.cos(lines2[j][0][1])
            b1 = np.sin(lines2[j][0][1])
            r1 = lines2[j][0][0]
            if b==0:#直线一斜率为0
                res_x = r
                res_y = (r1-res_x*a1)/b1
            elif b1 == 0:#直线二斜率为0
                res_x = r1
                res_y = (r-res_x*a)/b
            else:
                res_x = (r/b-r1/b1)/(a/b-a1/b1)
                res_y = (r-res_x*a)/b
            cv2.circle(image_old, (int(res_x * rate_x), int(res_y * rate_y)), 4, (255, 0, 0), 8)
            vec.append([int(res_x * rate_x), int(res_y * rate_y)])
    vec = np.array(vec,np.float32)
    print('vec:')
    print(vec)
    ReloadFourPoint(vec)

    plt.imshow(canny)
    plt.show()
    plt.imshow(result2)
    plt.show()
    plt.imshow(image_old)
    plt.show()
    canvas = [[0,0],[width,0],[width,height],[0,height]]
    canvas = np.array(canvas,np.float32)
    M = cv2.getPerspectiveTransform(vec,canvas)
    finalimg = cv2.warpPerspective(image_old, M, (0, 0))
    plt.imshow(finalimg)
    plt.show()
    final_res = finalimg[0:height,0:width]
    plt.imshow(final_res)
    plt.show()
    final_res = cv2.cvtColor(final_res, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./cutCropImage3.jpg", final_res)

if __name__ == '__main__':
    imagecut()
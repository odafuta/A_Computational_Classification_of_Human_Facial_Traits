import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

#create a file './ad_data' and add the files have the images of faces which can be animals or humans
#I use four animals faces each has 450 images, we can change as humans or add more images to improve the acc rate.
def imgLoad():
    images = []
    labels = []
    for id, animal in enumerate(['cat', 'dog', 'fox', 'tiger']):
        class_path = './af_data/' + str(animal)
        for i in os.listdir(class_path):
            img = Image.open(class_path + '/' + i).convert('L')
            img = img.resize((128, 128))
            imgArr = np.array(img).flatten()
            images.append(imgArr)
            labels.append(id)
    return np.array(images), np.array(labels)

#eigen true: fig the two main eigen; mean true: fig the mean fig; com: pca the number of the image, more large more information. 
def figPCA(x, eigen = False, mean = False, com = 100):
    pca = PCA(n_components = com)
    x_pca = pca.fit_transform(x)
    print("retain rate:", sum(pca.explained_variance_ratio_))
    eigen1 = pca.components_[0].reshape((128, 128))
    eigen2 = pca.components_[1].reshape((128, 128))
    if eigen == True:
        plt.subplot(1, 2, 1)
        plt.imshow(eigen1, cmap = 'gray')
        plt.title('eigen 1')
        plt.subplot(1, 2, 2)
        plt.imshow(eigen2, cmap = 'gray')
        plt.title('eigen 2')
        plt.show()
    if mean == True:
        labels = ['cat', 'dog', 'fox', 'tiger']
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            plt.scatter(
                x_pca[y == i, 0],
                x_pca[y == i, 1],
                label=labels[i],
                color=colors[i],
                alpha=0.6,
                edgecolor='k'
            )
        plt.show()
    return x_pca, pca

#components and C(punishment) are the adjustable parameters, by fine-tuning through accTest(), the aim is to achieve the highest accuracy. 
def xSVM(x, y, components = 110, C = 1.5, kernel = 'rbf', gamma = 'scale'):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
    x_train_pca, pca = figPCA(x_train, False, False, components)
    x_test_pca = pca.transform(x_test)
    svm = SVC(C = C, kernel = kernel, gamma = 'scale')
    svm.fit(x_train_pca, y_train)
    y_pred = svm.predict(x_test_pca)
    print("Acc rate:", accuracy_score(y_test, y_pred))
    joblib.dump(pca, f'./pca_{components}.pkl')
    joblib.dump(svm, f'./svm_c{C}_com{components}.pkl')
    return accuracy_score(y_test, y_pred), pca, svm

#The first three are the range and step size of components, then are them of C.
def accTest(lowN = 120, highN = 121, stepN = 1, lowC = 2, highC = 3, stepC = 1):
    name = str(lowN) + '_' + str(highN) + '_' + str(stepN)
    txt = []
    for i in range(lowN, highN, stepN):
        for j in range(int(lowC / stepC), int(highC / stepC), 1):
            a = xSVM(i, j * stepC)
            res = str(a) + ' > components = ' + str(i) + ', C = ' + str(j * stepC) + '\n'
            print(res)
            txt.append(res)
    with open('./res' + str(name) + '.txt', 'w') as file:
        file.writelines(txt)

#fig the svm.
def figSVM():
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 42, stratify = y)
    x_train_2d, pca_vis = figPCA(x_train, False, False, 2)
    svm_vis = SVC(kernel = 'rbf', C = 10, gamma = 'scale')
    svm_vis.fit(x_train_2d, y_train)
    h = 100
    x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
    y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    labels = ['cat', 'dog', 'fox', 'tiger']
    colors = ['red', 'green', 'blue', 'orange']
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for i in range(4):
        plt.scatter(
            x_train_2d[y_train == i, 0],
            x_train_2d[y_train == i, 1],
            c=colors[i],
            label=labels[i],
            edgecolor='k',
            alpha=0.8
        )
    plt.show()

#use the svm model to eva.
def imgEva(eva_path = './eva.jpg'):
    eva = Image.open(eva_path).convert('L')
    eva = eva.resize((128, 128))
    eva_array = np.array(eva).flatten().reshape(1, -1)
    eva_array_pca = pca.transform(eva_array)
    id = svm.predict(eva_array_pca)[0]
    animals = ['cat_like', 'dog_like', 'fox_like', 'tiger_like']
    print("eva res:", animals[id])

x, y = imgLoad()

#acc, pca, svm = xSVM(x, y) #for train

pca = joblib.load('./pca_110.pkl')
svm = joblib.load('./svm_c1.5_com110.pkl')

imgEva()

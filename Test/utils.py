import matplotlib.pyplot as plt


def imshow(X, y, subplot=(2, 5), figsize=(10, 4), suptitle="Images"):
    plt.figure(figsize=figsize)

    for index, (image, label) in enumerate(zip(X, y)):
        plt.subplot(subplot[0], subplot[1], index+1)
        plt.imshow(image.reshape(8, 8))
        plt.title(label)
        plt.axis('off')

    plt.suptitle(suptitle)
    plt.show()

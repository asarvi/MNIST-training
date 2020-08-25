from keras.datasets import mnist
import matplotlib.pyplot as plt


(train_features, train_labels), (test_features, test_labels) = mnist.load_data()
_, img_rows, img_cols =  train_features.shape

num_input_nodes = 28*28

fig = plt.figure(figsize=(8,3))
for i in range(2):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    features_idx = train_features[train_labels[:]==i,:]
    ax.set_title("Num: " + str(i))
    plt.imshow(features_idx[1], cmap="gray")
plt.show()

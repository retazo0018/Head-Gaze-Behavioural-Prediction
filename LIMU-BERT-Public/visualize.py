'''Visualize Results from Results of LIMU BERT'''

from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class Visualize():
    def __init__(self,preds,labels):
        self.preds = preds
        self.labels = labels

    def plot3D(self):
        preds_reshaped = self.preds.reshape(self.preds.shape[0], -1)
        labels_reshaped = self.labels.reshape(self.labels.shape[0], -1)
        pca = PCA(n_components=3)
        preds_3d = pca.fit_transform(preds_reshaped)
        labels_3d = pca.transform(labels_reshaped)

        # Plotting the results
        fig = plt.figure(figsize=(12, 6))

       # Output Vectors
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(preds_3d[:, 0], preds_3d[:, 1], preds_3d[:, 2], color='blue', alpha=0.5)
        ax1.set_title('Output Vectors (3D PCA)')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.set_zlabel('PCA Component 3')

        # Training Vectors
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(labels_3d[:, 0], labels_3d[:, 1], labels_3d[:, 2], color='green', alpha=0.5)
        ax2.set_title('Training Vectors (3D PCA)')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        ax2.set_zlabel('PCA Component 3')

        plt.show()


    def plot2D(self):
        preds_reshaped = self.preds.reshape(self.preds.shape[0], -1)
        labels_reshaped = self.labels.reshape(self.labels.shape[0], -1)
        pca = PCA(n_components=2)
        preds_2d = pca.fit_transform(preds_reshaped)
        labels_2d = pca.transform(labels_reshaped)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(preds_2d[:, 0], preds_2d[:, 1], color='blue', alpha=0.5, label='Output Vectors')
        plt.title('Output Vectors (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.subplot(1, 2, 2)
        plt.scatter(labels_2d[:, 0], labels_2d[:, 1], color='green', alpha=0.5, label='Training Vectors')
        plt.title('Training Vectors (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.tight_layout()
        plt.show()

    def plot3DLine(self,n=20):
        #Randomly choosing n vectors from the training data:
        indexes = np.random.choice(self.preds.shape[0],n, replace = False)
        preds_3d = self.preds[indexes,:]
        labels_3d = self.labels[indexes,:]

        fig = plt.figure(figsize=(12, 6))

       # Output Vectors
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        for sequence in preds_3d:
            ax1.plot(sequence[:, 0], sequence[:, 1], sequence[:, 2])
        ax1.set_title('Generated Data')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # Training/Test Vectors
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        for sequence in labels_3d:
            ax2.plot(sequence[:, 0], sequence[:, 1], sequence[:, 2])
        ax2.set_title('Actual Data')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.show()

        
    
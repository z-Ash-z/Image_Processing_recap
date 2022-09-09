import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class ImageAlgo:

    def __init__(self, images_path, results_path) -> None:
        self.images_path = images_path
        self.results_path = results_path

    def readImage(self, image):
        """
        Reads and stores the image from the path

        Args:
            image: the name of the image file - "image.format"
        """
        self.image = Image.open(self.images_path + image)
        self.image_matrix = np.array(self.image)
        self.image_rows, self.image_columns, _ = self.image_matrix.shape

    def showImage(self, window_name = "Stored Image"):
        """
        Shows the image stored
        """
        self.image.show(window_name)

    def plotScanLine(self, linenumber = 0, result_name = None):
        """
        Plots the RBG channels from the selected scanline in the stored image 

        Args:
            linenumber: The line number to scan and plot. Defaults to 0.
            result_name: If a result name is given then stores it with given name in the results path. Defaults to None.
        """
        red_channel = self.image_matrix[linenumber, :, 0]
        blue_channel = self.image_matrix[linenumber, :, 1]
        green_channel = self.image_matrix[linenumber, :, 2]

        x_axis = np.arange(0, self.image_columns)

        plt.title(f'RGB plot along scanline {linenumber}')
        plt.plot(x_axis, red_channel, 'r-', label = 'Red Channel')
        plt.plot(x_axis, blue_channel, 'b-', label = 'Blue Channel')
        plt.plot(x_axis, green_channel, 'g-', label = 'Green Channel')
        plt.legend()

        if result_name != None:
            plt.savefig(self.results_path + result_name)
            
        plt.show()   
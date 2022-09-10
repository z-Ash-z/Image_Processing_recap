import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class ImageAlgo:

    RED_CHANNEL_INDEX = 0
    BLUE_CHANNEL_INDEX = 2
    GREEN_CHANNEL_INDEX = 1
    CHANNELS = 3

    def __init__(self, images_path : str, results_path : str) -> None:
        """
        Constructor of the ImageAlgo.

        Args:
            images_path: The path for the input images.
            results_path: The path for the results.
        """
        self.images_path = images_path
        self.results_path = results_path

    def readImage(self, image : str):
        """
        Reads and stores the image from the path.

        Args:
            image: the name of the image file - "image.format".
        """
        self.image = Image.open(self.images_path + image)
        self.image_matrix = np.array(self.image, dtype = np.uint8)
        self.image_rows, self.image_columns, _ = self.image_matrix.shape

    def showImage(self, window_name : str = "Stored Image"):
        """
        Shows the image stored.
        """
        self.image.show(window_name)

    def plotScanLine(self, linenumber : int = 0, result_name : str = None) -> None:
        """
        Plots the RBG channels from the selected scanline in the stored image .

        Args:
            linenumber: The line number to scan and plot. Defaults to 0.
            result_name: If a result name is given then stores it with given name in the results path. Defaults to None.
        """
        red_channel = self.image_matrix[linenumber, :, self.RED_CHANNEL_INDEX]
        blue_channel = self.image_matrix[linenumber, :, self.BLUE_CHANNEL_INDEX]
        green_channel = self.image_matrix[linenumber, :, self.GREEN_CHANNEL_INDEX]

        x_axis = np.arange(0, self.image_columns)

        plt.title(f'RGB plot along scanline {linenumber}')
        plt.plot(x_axis, red_channel, 'r-', label = 'Red Channel')
        plt.plot(x_axis, blue_channel, 'b-', label = 'Blue Channel')
        plt.plot(x_axis, green_channel, 'g-', label = 'Green Channel')
        plt.legend()

        if result_name != None:
            plt.savefig(self.results_path + result_name)

        plt.show()

    def stackImages(self, result_name : str = None) -> None:
        """
        Stacks the red, blue and green channels vertically into a single image

        Args:
            result_name: Saves the generated image with the given name, if the given name is not None. Defaults to None.
        """
        new_image = np.zeros((3*self.image_rows, self.image_columns, self.CHANNELS), dtype = np.uint8) 

        for i in range(self.CHANNELS):
            new_image[(self.image_rows * i) : (self.image_rows * (i + 1)), :, i] = self.image_matrix[:, :, i]

        stacked_image = Image.fromarray(new_image)
        
        if result_name != None:
            stacked_image.save(self.results_path + result_name)
        
        stacked_image.show("Result")

    def swapChannels(self, result_name : str = None) -> None:
        """
        Swaps the red and green channel of the input image

        Args:
            result_name: Saves the generated image with the given name, if the given name is not None. Defaults to None.
        """
        swapped_image = self.image_matrix
        swapped_image[:, :, self.RED_CHANNEL_INDEX], swapped_image[:, :, self.GREEN_CHANNEL_INDEX] = \
            swapped_image[:, :, self.GREEN_CHANNEL_INDEX], swapped_image[:, :, self.RED_CHANNEL_INDEX]

        swapped_image = Image.fromarray(swapped_image)

        if result_name != None:
            swapped_image.save(self.results_path + result_name)

        swapped_image.show("After swapping")

    def convertToGray(self, result_name : str = None) -> None:
        """
        Converts the given image to grayscale. Reference for math: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

        Args:
            result_name: Saves the generated image with the given name, if the given name is not None. Defaults to None.
        """
        gray_image = np.zeros_like((self.image_rows, self.image_columns), dtype = np.uint8)
        gray_image = self.image_matrix[:, :, self.RED_CHANNEL_INDEX] * 0.2126 + \
            self.image_matrix[:, :, self.GREEN_CHANNEL_INDEX] * 0.7152 + \
            self.image_matrix[:, :, self.BLUE_CHANNEL_INDEX] * 0.0722
        gray_image = np.uint8(gray_image)

        gray_image = Image.fromarray(gray_image)

        if result_name != None:
            gray_image.save(self.results_path + result_name)

        gray_image.show("Gray Image")

    def channelAverage(self, result_name : str = None) -> None:
        """
        Averages the three channels to a single channel.

        Args:
            result_name: Saves the generated image with the given name, if the given name is not None. Defaults to None.
        """
        channel_average_image = np.zeros_like((self.image_rows, self.image_columns), dtype = np.uint8)
        channel_average_image = (self.image_matrix[:, :, self.RED_CHANNEL_INDEX] + \
            self.image_matrix[:, :, self.GREEN_CHANNEL_INDEX] + \
            self.image_matrix[:, :, self.BLUE_CHANNEL_INDEX]) / 3
        channel_average_image = np.uint8(channel_average_image)

        channel_average_image = Image.fromarray(channel_average_image)

        if result_name != None:
            channel_average_image.save(self.results_path + result_name)

        channel_average_image.show("Channel Average")
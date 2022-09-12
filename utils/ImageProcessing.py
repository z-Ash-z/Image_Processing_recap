import os
import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class ImageAlgo:

    RED_CHANNEL_INDEX = 0
    BLUE_CHANNEL_INDEX = 2
    GREEN_CHANNEL_INDEX = 1

    def __init__(self, images_path : str, results_path : str, display : bool = False) -> None:
        """
        Constructor of the ImageAlgo.

        Args:
            images_path : The path for the input images.
            results_path: The path for the results.
            display     : Sets the flag to display the results, display if true. Defaults to False 
        """
        current_path = os.path.dirname(__name__)
        self.images_path = os.path.join(current_path, images_path)
        self.results_path = os.path.join(current_path, results_path)
        if os.path.exists(self.results_path) != True:
            os.mkdir(self.results_path)
        self.display = display
        self.mask_locations_flag = False
        self.gray_image_flag = False

    def readImage(self, image : str):
        """
        Reads and stores the image from the path.

        Args:
            image: the name of the image file - "image.format".
        """
        self.image = Image.open(self.images_path + image)
        self.image_matrix = np.array(self.image, dtype = np.uint8)
        self.image_rows, self.image_columns, self.channels = self.image_matrix.shape

    def showImage(self, window_name : str = "Stored Image"):
        """
        Shows the image stored.
        """
        self.image.show(window_name)

    def plotScanLine(self, linenumber : int = 0, result_name : str = None, show_image : bool = False) -> None:
        """
        Plots the RBG channels from the selected scanline in the stored image .

        Args:
            linenumber  : The line number to scan and plot. Defaults to 0.
            result_name : If a result name is given then stores it with given name in the results path. Defaults to None.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        red_channel = copy.deepcopy(self.image_matrix[linenumber, :, self.RED_CHANNEL_INDEX])
        blue_channel = copy.deepcopy(self.image_matrix[linenumber, :, self.BLUE_CHANNEL_INDEX])
        green_channel = copy.deepcopy(self.image_matrix[linenumber, :, self.GREEN_CHANNEL_INDEX])

        x_axis = np.arange(0, self.image_columns)

        plt.title(f'RGB plot along scanline {linenumber}')
        plt.plot(x_axis, red_channel, 'r-', label = 'Red Channel')
        plt.plot(x_axis, blue_channel, 'b-', label = 'Blue Channel')
        plt.plot(x_axis, green_channel, 'g-', label = 'Green Channel')
        plt.legend()

        if result_name != None:
            plt.savefig(self.results_path + result_name)

        if show_image or self.display:
            plt.show()

    def stackImages(self, result_name : str = None, color : bool = True, show_image : bool = False) -> None:
        """
        Stacks the red, blue and green channels vertically into a single image

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            color       : When True stacks the images with color else in grayscale. Defaults to True.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        if color:
            new_image = np.zeros((3*self.image_rows, self.image_columns, self.channels), dtype = np.uint8) 

            for i in range(self.channels):
                new_image[(self.image_rows * i) : (self.image_rows * (i + 1)), :, i] = copy.deepcopy(self.image_matrix[:, :, i])
        else:
            new_image = np.zeros((3*self.image_rows, self.image_columns), dtype = np.uint8) 

            for i in range(self.channels):
                new_image[(self.image_rows * i) : (self.image_rows * (i + 1)), :] = copy.deepcopy(self.image_matrix[:, :, i])

        stacked_image = Image.fromarray(new_image)
        
        if result_name != None:
            stacked_image.save(self.results_path + result_name)
        
        if show_image or self.display:
            stacked_image.show("Result")

    def swapChannels(self, result_name : str = None, show_image : bool = False) -> None:
        """
        Swaps the red and green channel of the input image

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        swapped_image = copy.deepcopy(self.image_matrix)
        swapped_image[:, :, self.RED_CHANNEL_INDEX], swapped_image[:, :, self.GREEN_CHANNEL_INDEX] = \
            swapped_image[:, :, self.GREEN_CHANNEL_INDEX], swapped_image[:, :, self.RED_CHANNEL_INDEX]

        swapped_image = Image.fromarray(swapped_image)

        if result_name != None:
            swapped_image.save(self.results_path + result_name)

        if show_image or self.display:
            swapped_image.show("After swapping")

    def convertToGray(self, result_name : str = None, show_image : bool = False) -> None:
        """
        Converts the given image to grayscale. Reference for math: https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        gray_image = np.zeros_like((self.image_rows, self.image_columns), dtype = np.uint8)
        gray_image = self.image_matrix[:, :, self.RED_CHANNEL_INDEX] * 0.2126 + \
            self.image_matrix[:, :, self.GREEN_CHANNEL_INDEX] * 0.7152 + \
            self.image_matrix[:, :, self.BLUE_CHANNEL_INDEX] * 0.0722
        self.gray_image_matrix = np.uint8(gray_image)

        gray_image = Image.fromarray(self.gray_image_matrix)

        if result_name != None:
            gray_image.save(self.results_path + result_name)

        if show_image or self.display:
            gray_image.show("Gray Image")

        self.gray_image_flag = True

    def channelAverage(self, result_name : str = None, show_image : bool = False) -> None:
        """
        Averages the three channels to a single channel.

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        channel_average_image = np.zeros_like((self.image_rows, self.image_columns), dtype = np.uint8)
        channel_average_image = (self.image_matrix[:, :, self.RED_CHANNEL_INDEX] + \
            self.image_matrix[:, :, self.GREEN_CHANNEL_INDEX] + \
            self.image_matrix[:, :, self.BLUE_CHANNEL_INDEX]) / 3
        channel_average_image = np.uint8(channel_average_image)

        channel_average_image = Image.fromarray(channel_average_image)

        if result_name != None:
            channel_average_image.save(self.results_path + result_name)

        if show_image or self.display:
            channel_average_image.show("Channel Average")

    def negativeImage(self, result_name : str = None, show_image : bool = False) -> None:
        """
        Creates the negative of the grayscaled image

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        if not self.gray_image_flag:
            self.convertToGray()
        negative_image = 255 - copy.deepcopy(self.gray_image_matrix)

        negative_image = Image.fromarray(negative_image)

        if result_name != None:
            negative_image.save(self.results_path + result_name)

        if show_image or self.display:
            negative_image.show("Negative Image")

    def cropRotateStack(self, result_name : str = None, clockwise : bool = False, show_image : bool = False) -> None:
        """
        Crops a 372 x 372 image from the read image, rotates it by 90 degrees 3 times and stacks all the four images horizontally.

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            clockwise   : If true rotates the crop to clockwise direction. Defaults to False.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        row_center = self.image_rows // 2
        column_center = self.image_columns // 2
        crop_size = 372

        cropped_image = np.zeros((crop_size, crop_size, self.channels), dtype = np.uint8)
        stacked_image = np.zeros((crop_size, crop_size * 4, self.channels), dtype = np.uint8)
        
        cropped_image = copy.deepcopy(self.image_matrix[row_center : row_center + crop_size, column_center : column_center + crop_size, :])
        
        rotate_direction = (0, 1)
        if clockwise:
            rotate_direction = (1, 0)

        for i in range(4):
            stacked_image[:, crop_size * i : crop_size * (i + 1), :] = cropped_image
            cropped_image = np.rot90(cropped_image, axes = rotate_direction)

        stacked_image = Image.fromarray(stacked_image)

        if result_name != None:
            stacked_image.save(self.results_path + result_name)

        if show_image or self.display:
            stacked_image.show()

    def masking(self, result_name : str = None, threshold : int = 127, show_image : bool = False) -> None:
        """
        When a pixel value is greater than threshold in any of the channels the corresponding value is changed to 255.

        Args:
            result_name : Saves the generated image with the given name, if the given name is not None. Defaults to None.
            threshold   : The value above which the pixel values are changed to 255. Defaults to 127.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        masked_image = np.zeros_like(self.image_matrix, dtype = np.uint8)

        self.mask_locations = np.where(self.image_matrix > threshold)
        masked_image[self.mask_locations] = 255
        
        masked_image = Image.fromarray(masked_image)

        if result_name != None:
            masked_image.save(self.results_path + result_name)

        if show_image or self.display:
            masked_image.show()
        
        self.mask_locations_flag = True

    def mean_calculator(self, threshold : int = 127) -> None:
        """
        The mean of the pixel values that are above the threshold.
        The mean is calculated for each channel seperately.

        Args:
            threshold: The value above which the pixel values are changed to 255. Defaults to 127.
        """
        if not self.mask_locations_flag:
            self.masking(threshold = threshold)

        red_values = []
        green_values = []
        blue_values = []

        for i in range(len(self.mask_locations[0])):
            if self.mask_locations[2][i] == self.RED_CHANNEL_INDEX:
                red_values.append(self.image_matrix[self.mask_locations[0][i], self.mask_locations[1][i], self.mask_locations[2][i]])
            if self.mask_locations[2][i] == self.GREEN_CHANNEL_INDEX:
                green_values.append(self.image_matrix[self.mask_locations[0][i], self.mask_locations[1][i], self.mask_locations[2][i]])
            if self.mask_locations[2][i] == self.BLUE_CHANNEL_INDEX:
                blue_values.append(self.image_matrix[self.mask_locations[0][i], self.mask_locations[1][i], self.mask_locations[2][i]])

        print(f'The mean values in the mask:-')
        print(f'Red Channel\t: {round(np.mean(red_values), 4)}')
        print(f'Green Channel\t: {round(np.mean(green_values), 4)}')
        print(f'Blue Channel\t: {round(np.mean(blue_values), 4)}')

    def maximumValueWindow(self, result_name : str = None, window_size = 5, show_image : bool = False) -> None:
        """
        Creates a binary image from the grayscaled version of the image. The grayscaled image is converted by finding the maximum value
        in each window and setting that as 255.

        Args:
            result_name: Saves the generated image with the given name, if the given name is not None. Defaults to None.
            window_size: The size of the window in the grayscaled image that you want to find the maximum value from. Defaults to 5.
            show_image  : If set to True, shows the image in a window after computing. Defaults to False.
        """
        if not self.gray_image_flag:
            self.convertToGray()
        binary_image = np.zeros_like(self.gray_image_matrix, dtype = np.uint8)

        for row in range(self.image_rows - window_size):
            for column in range(self.image_columns - window_size):
                window = self.gray_image_matrix[row : row + window_size, column : column + window_size]
                max_value = np.max(window)
                max_value_indeces = np.where(window == max_value)
                binary_image[row : row + window_size, column : column + window_size][max_value_indeces] = 255
        
        binary_image = Image.fromarray(binary_image)

        if result_name != None:
            binary_image.save(self.results_path + result_name)

        if show_image or self.display:
            binary_image.show()
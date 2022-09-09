from utils.ImageProcessing import ImageAlgo

if __name__ == '__main__':
    """
    The main code for all the problems in assignment-0 in CMSC733
    Author: Aneesh Chodisetty
    Date created: 09/09/2022
    """
    # Setting the input and the results path.
    images_path = 'source/'
    results_path = 'results/'
    im = ImageAlgo(images_path, results_path)

    # Reading the given image.
    im.readImage('iribefront.jpg')
    
    # Problem - 1, plotting the RGB values along the scanline.
    im.plotScanLine(250, '1_scanline.png')

    # Problem - 2, Stack the R, G, B channels of the image vertically.
    im.stackImages('2_concat.png')
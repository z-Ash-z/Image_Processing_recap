from utils.ImageProcessing import ImageAlgo

if __name__ == '__main__':
    """
    The main code for all the problems in assignment-0
    Author: Aneesh Chodisetty
    """
    images_path = 'source/'
    results_path = 'results/'

    im = ImageAlgo(images_path, results_path)
    im.readImage('iribefront.jpg')
    # im.showImage()

    # im.plotScanLine(250)
    # print(im.image_matrix.shape)
    # print(len(im.image_matrix[250, :, 0]))
    im.plotScanLine(250, '1_scanline.png')

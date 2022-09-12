# Image Processing basics
This is the work created for the submission of Homework-0 for CMSC733 - Computer Processing of Pictorial Information.

Run `Assignment_0.py` for solving the following problem statements.
1. Plot the R, G, B values along the scanline on the 250th row of the image.
    -  Save your plot as “1_scanline.png”.

2. Stack the R, G, B channels of the image vertically.
    - Save the composite image as “2_concat.png”.

3. Load the input color image and swap its red and green color channels.
    - Save the image as “3_swapchannel.png”.

4. Convert the input color image to a grayscale image.
    - Save the grayscale image as “4_grayscale.png”.

5. Take the R, G, B channels of the image. Compute an average over the three channels. Note that you may need to do the necessary typecasting (uint8 and double) to avoid overflow.
    - Save the averaged image as “5_average.png”.

6. Take the grayscale image in (4), obtain the negative image (i.e., mapping 255 to 0 and 0 to 255).
    - Save the image as “6_negative.png”.

7. First, crop the original image into a squared image of size 372 x 372. Then, rotate the image by 90, 180, and 270 degrees and stack the four images (0, 90, 180, 270 degreess) horizontally.
    - Save the image as “7_rotation.png”.

8. Create another image with the same size as the image. First, initialize this image as zero everywhere. Then, for each channel, set the pixel values as 255 when the corresponding pixel values in the image are greater than 127.
    - Save the image as “8_mask.png”.

9. Report the mean R, G, B values for those pixels marked by the mask in (8).

10. Take the grayscale image in (3). Create and initialize another image as all zeros. For each 5 x 5 window in the grayscale image, find out the maximum value and set the pixels with the maximum value in the 5x5 window as 255 in the new image.
    - Save the result image as “10_nonmax.png”.

### The image used is:
![Iribe](source\iribefront.jpg)

---

## Python dependenceies
* os
* copy
* numpy
* PIL
* matplotlib
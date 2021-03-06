# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
from tqdm import tqdm
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# Define some constants 
file_name = "Boom"  # Filename of the bigpicture
NUMBER_PIXPICS = 33000     # Amount of small images that will be used to create the big picture
MAX_IM = 150      # Maximum amount of images that will be iterated over to try and recreate the bigpicture
MOST_USED_THRESHOLD = 0.12 # Determines a treshold to show the most used images
error_type = 'Mean'     # Choose the error type (Mean, Euclidian, CIE2000)     TODO: Use this to select wich error function is used
PIXEL_SKIP = 3  # Tells in how many pieces we divide dim pix, to speed up the CIE2000 error calculation

# Main function of the program
def main():

    # Read in the main picture and get needed parameters                      TODO: dont work with fixed pathways
    parentdir = "D://DocumentenDschijf//3de bach//PicturestoCollage//"
    main_pic_dir = parentdir + "BigPictures//" + file_name + ".jpg"
    pixel_pics_dir = parentdir + "PixelPictures5//"
    save_dir = parentdir + "Finished//"

    mainpicture = read_picture(main_pic_dir)
    height1, width1 = len(mainpicture), len(mainpicture[0])

    # Determine size of pixel pictures
    dim_pix, amount_height, amount_width = size_pixel_pic(height1, width1)

    # Determine a precision factor that will say for which pixels the CIE2000 is evaluated
    PRECISION_FACTOR = int(dim_pix/PIXEL_SKIP)

    # Crop the main picture to fit perfect squares of dim_pix init
    mainpicture = crop(mainpicture, height1, width1, amount_height, amount_width, dim_pix)

    # Create empty image for signal reconstruction
    reconstructed = create_matrix(amount_height * dim_pix, amount_width * dim_pix, [np.uint8(0), np.uint8(0), np.uint8(0)])

    # Create Current error matrix
    current_error = create_matrix(amount_height, amount_width, None)

    # Create image counter
    image_count = create_matrix(amount_height, amount_width, None)

    # Create a matrix containing the mean values of the main picture (only used for the mean error)   TODO: Only do this when the error is "mean"
    ''' mean_main_picture = []
    for h in range(height1):
        list = []
        for w in range(width1):
            mean_rgb = np.sum(np.sum(mainpicture[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix], axis = 0), axis = 0)/(dim_pix*dim_pix)
            list.append(mean_rgb)
        mean_main_picture.append(list)
    mean_main_picture = np.array(mean_main_picture) '''

    # Read in the pixel pictures, resize, calculate error and associate with correct spot
    for im_num in tqdm(range(1, MAX_IM+1)):

        # Read in the small image 
        pixel_pic = read_picture(pixel_pics_dir + "Pixel_Im ("+ str(im_num) + ").jpg")

        # Cut the image to correct ratio (square)
        height2, width2 = len(pixel_pic), len(pixel_pic[0])
        if height2 >= width2:
            diff = height2-width2
            pixel_pic = pixel_pic[diff//2:height2-diff//2,:]
        else:
            diff = width2-height2
            pixel_pic = pixel_pic[:,diff//2:width2-diff//2]

        # Resize the image to dim_pix by dim_pix
        pixel_pic = resize(pixel_pic, dim_pix)

        # Calculate error (choose between mean or euclidian)                                                TODO: Make this cleaner such that it fits for every error function
        #error = calculate_error_mean(mean_main_picture, pixel_pic, amount_height, amount_width, dim_pix)
        #error = calculate_error_euclidian(mainpicture, pixel_pic, amount_height, amount_width, dim_pix)
        #error = calculate_error_perception(amount_height, amount_width, pixel_pic, mainpicture, dim_pix)
        error = calculate_error_cie2000(amount_height, amount_width, pixel_pic, mainpicture, dim_pix, PRECISION_FACTOR)

        # Reconstruct the image, and update image counter and current error
        reconstructed, current_error, image_count = reconstruct(amount_height, amount_width, error, current_error, dim_pix, reconstructed, image_count, pixel_pic, im_num)

    # Quantise the images that are not used
    zero_images, counter, x_axis = create_zero_images(image_count)
    
    # Quantise the imagess that are used a lot
    max_images = create_max_images(counter, x_axis)

    # Print out some of the results about used images and different amount of images
    print("Images that are not used = ", zero_images)
    print("Images that are used a lot = ", max_images)
    print("Total amount of different images = ", MAX_IM - len(zero_images))

    # Calculate the total error
    tot_error = int(np.sum(current_error.flatten()))

    # Save the new image with (almost) the same resoltion as the original
    save_name = save_dir + file_name + "_collage_AmountOfImages_" + str(NUMBER_PIXPICS) + "DifferentImages_" + str(MAX_IM - len(zero_images)) + "Total_Error_" + str(tot_error) + ".jpg"
    save(save_name, reconstructed)

    # Save the data of image_count so that later on a picture with higher resoltution could be made   TODO: Write a small script that can create this high res image
    np.savetxt(file_name + ".csv", image_count, delimiter=",")

    # Visualise the current error
    plot_matrix(current_error)

    # Visualise the amount of times each image is used
    plot(counter, x_axis)

# TODO: Clean up the description of functions and specify the meaning of used arguments as folows using a docstring:
    """[summary]

    Returns:
        [type]: [description]
    """

# Function that will read in the correct picture from the correct location
def read_picture(dir):
    return plt.imread(dir)


# Function that will crop the picture:
def resize(picture, dim_pix):
    return cv2.resize(picture, (dim_pix,dim_pix), interpolation = cv2.INTER_AREA)


# Function that will resize the picture:
def crop(picture, height, width, amount_height, amount_width, dim_pix):
    diff_height = height-dim_pix*amount_height
    diff_width = width-dim_pix*amount_width
    return picture[diff_height//2:dim_pix*amount_height+diff_height//2, diff_width//2 :dim_pix*amount_width+diff_width//2]


# Function that will calculate the error: (sum of the euclidian distances between rgb values of the main picture and the "pixel_pic")
def calculate_error_euclidian(picture1, picture2, height, width, dim_pix):
    error = []
    for h in range(height):
        error2 = []
        for w in range(width):
            pixel_diff = np.array(picture2-picture1[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix])
            error1 = []
            for pain in pixel_diff:
                error1.append(np.sum(np.sqrt(np.sum(np.array(pain**2), axis = 1))))
            error2.append(np.sum(error1))
        error.append(error2)
    return np.array(error)


# Function that will calculate the error: (sum of the euclidian distances between rgb values of the main picture and the "pixel_pic")
def calculate_error_mean(picture1, picture2, height, width, dim_pix):
    error = []
    for h in range(height):
        error2 = []
        for w in range(width):
            mean_rgb_picture1 = picture1[h][w]
            mean_rgb_picture2 = np.sum(np.sum(picture2, axis = 0), axis = 0)/(dim_pix*dim_pix)
            error_value = np.abs(np.sum(mean_rgb_picture1) - np.sum(mean_rgb_picture2))
            error2.append(error_value)
        error.append(error2)
    return np.array(np.float16(error))

# Function that will calculate the error based on the square root formula (ref: wikipedia Color Differences)
def calculate_error_perception(height, width, pixel_pic, mainpicture, dim_pix):
    error = []
    for h in range(height):
        error2 = []
        for w in range(width):
            pixel_diff = np.array(pixel_pic-mainpicture[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix])
            pixel_add = np.array(pixel_pic+mainpicture[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix])
            error1 = []
            for pain, pain2 in zip(pixel_diff, pixel_add):
                for worse, worse2 in zip(pain, pain2):
                    dR, dG, dB = worse[0] , worse[1], worse[2]
                    r = worse2[0]/2+1
                    error1.append(np.sqrt((2+r/256)*dR**2 + 4*dG**2 + (2+(256-r)/r)*dB**2 ))
            error2.append(np.sum(error1))
        error.append(error2)
    return np.array(error)

# Function that will calculate the error based on the CIE2000 measure
def calculate_error_cie2000(height, width, pixel_pic, mainpicture, dim_pix, PRECISION_FACTOR):
    error = []
    for h in range(height):
        error1 = []
        for w in range(width):
            main = mainpicture[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix]
            error2 = []
            for i in range(len(main)//PRECISION_FACTOR):
                error3 = []
                for j in range(len(main[0])//PRECISION_FACTOR):
                    rgb1 = sRGBColor(main[PRECISION_FACTOR*i][PRECISION_FACTOR*j][0], main[PRECISION_FACTOR*i][PRECISION_FACTOR*j][1], main[PRECISION_FACTOR*i][PRECISION_FACTOR*j][2])
                    rgb2 = sRGBColor(pixel_pic[PRECISION_FACTOR*i][PRECISION_FACTOR*j][0], pixel_pic[PRECISION_FACTOR*i][PRECISION_FACTOR*j][1], pixel_pic[PRECISION_FACTOR*i][PRECISION_FACTOR*j][2])
                    lab1 = convert_color(rgb1, LabColor)
                    lab2 = convert_color(rgb2, LabColor)
                    delta_e = delta_e_cie2000(lab1, lab2)
                    error3.append(delta_e)
                error2.append(sum(error3))
            error1.append(sum(error2))
        error.append(error1)
    return np.array(error)


# Function that will create matrices of given size and with given but fixed values init
def create_matrix(height, width, values):
    matrix = []

    for h in range(height):
        list = []
        for w in range(width):
            list.append(values)
        matrix.append(list)

    return np.array(matrix) #TODO: dubbel werk maar het gwn direct aan als np array en gebruik dan


# Function that determines size of pixel pics
def size_pixel_pic(height, width):
    dim_pix = int(round(np.sqrt((height*width)/NUMBER_PIXPICS)))
    amount_height = height//dim_pix
    amount_width = width//dim_pix
    return dim_pix, amount_height, amount_width


# Function that reconstructs the image
def reconstruct(height, width, error, current_error, dim_pix, reconstructed, image_count, pixel_pic, im_num):
    for h in range(height):
        for w in range(width):
            if current_error[h][w] == None or error[h][w] < current_error[h][w]:
                reconstructed[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix] = pixel_pic
                current_error[h][w] = error[h][w]
                image_count[h][w] = im_num

    return reconstructed, current_error, image_count


# Function that looks wich images are not used
def create_zero_images(image_count):
    zero_images = []
    x_axis = []
    counter = []

    for i in range(1, MAX_IM+1):
        x_axis.append(i)
        counter.append(np.count_nonzero(image_count == i))
        if np.count_nonzero(image_count == i) == 0:
            zero_images.append(i)
    return zero_images, counter, x_axis


# Function that looks for images that are used a lot
def create_max_images(counter, x_axis):
    max_images = []

    for j in range(len(counter)):
        if counter[j] >= MOST_USED_THRESHOLD * max(counter): 
            max_images.append(x_axis[j])
    return max_images


# Function to save the results
def save(dir, data):
    plt.imsave(dir, data)


# Function to visualise matrix data
def plot_matrix(data):
    plt.figure()
    plt.imshow(np.array(data, dtype=float))
    plt.colorbar()


# Function to visualise normal data in a 1d list
def plot(y_data, x_data):
    plt.figure()
    plt.plot(x_data, y_data, 'o')
    plt.show()


# Activate when this is the main program, call the main function
if __name__ == '__main__':
    main()



# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
from tqdm import tqdm

# Set options like file name,...
file_name = "DemonSlayer"  # Filename of the bigpicture
NUMBER_PIXPICS = 2000     # Amount of small images that will be used to create the big picture
MAX_IM = 10             # Maximum amount of images that will be iterated over to try and recreate the bigpicture
RESOLUTION_FACTOR = 2     # The upscaling factor to increase the size of the final image 
MOST_USED_THRESHOLD = 0.12 # Determines a treshold to show the most used images

# Set file locations
parentdir = "D://DocumentenDschijf//3de bach//PicturestoCollage//"
main_pic_dir = parentdir + "BigPictures//" + file_name + ".jpg"     #TODO: zie tutorial manu 
pixel_pics_dir = parentdir + "PixelPictures5//"
save_dir = parentdir + "Finished//"

# Read in the big picture and identify the needed parameters
mainpicture = plt.imread(main_pic_dir)
height1, width1 = len(mainpicture), len(mainpicture[0])

# Determine size of pixel pictures
dim_pix = int(np.sqrt((height1*width1)/NUMBER_PIXPICS))
amount_height = height1//dim_pix
amount_widht = width1//dim_pix

# Resize the main picture to fit perfect squares of dim_pix init
diff_height = height1-dim_pix*amount_height
diff_width = width1-dim_pix*amount_widht
mainpicture = mainpicture[diff_height//2:dim_pix*amount_height+diff_height//2, diff_width//2 :dim_pix*amount_widht+diff_width//2] #TODO: kortere lijn door op volgende 'lijn' te zetten

# Create 'empty' image for recontstruction
reconstructed = []

for h in range(amount_height*dim_pix):
    list = []
    for w in range(amount_widht*dim_pix):
        list.append([np.uint8(0), np.uint8(0), np.uint8(0)])
    reconstructed.append(list)

reconstructed = np.array(reconstructed) #TODO: dubbel werk maar het gwn direct aan als np array en gebruik dan

# Create Current error matrix and image counter
current_error = []
image_count = []

for h in range(amount_height):
    list = []
    for w in range(amount_widht):
        list.append(None)
    current_error.append(list)
    image_count.append(list)

current_error = np.array(current_error)
image_count = np.array(image_count)

# Read in the pixel pictures, resize, calculate error and associate with correct spot
for im_num in tqdm(range(1, MAX_IM+1)):

    # Read the small image in
    pixel_pic =  plt.imread(pixel_pics_dir + "Pixel_Im ("+ str(im_num) + ").jpg")  #TODO: zie tweede tutorial van manu

    # Cut the image to correct ratio (square)
    height2, width2 = len(pixel_pic), len(pixel_pic[0])
    if height2 >= width2:
        diff = height2-width2
        pixel_pic = pixel_pic[diff//2:height2-diff//2,:]
    else:
        diff = width2-height2
        pixel_pic = pixel_pic[:,diff//2:width2-diff//2]

    # Resize the image to dim_pix by dim_pix 
    pixel_pic = cv2.resize(pixel_pic, (dim_pix,dim_pix), interpolation = cv2.INTER_AREA)

    # Calculate error based on the sum of the euclidian distances between rgb values of the main picture and the "pixel_pic"
    error = []
    for h in range(amount_height):
        error2 = []
        for w in range(amount_widht):
            pixel_diff = np.array(pixel_pic-mainpicture[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix])
            error1 = []
            for pain in pixel_diff:
                error1.append(np.sum(np.sqrt(np.sum(np.array(pain**2), axis = 1))))
            error2.append(np.sum(error1))
        error.append(error2)
    error = np.array(error)

    # Reconstruct the image
    for h in range(amount_height):
        for w in range(amount_widht):
            if current_error[h][w] == None or error[h][w] < current_error[h][w]:
                reconstructed[h*dim_pix:(h+1)*dim_pix, w*dim_pix:(w+1)*dim_pix] = pixel_pic
                current_error[h][w] = error[h][w]
                image_count[h][w] = im_num


# Quantise the used images, define not used and most used
counter = []
x_axis = []
zero_images = []
max_images = []

for i in range(1, MAX_IM+1):
    x_axis.append(i)
    counter.append(np.count_nonzero(image_count == i))
    if np.count_nonzero(image_count == i) == 0:
        zero_images.append(i)

for j in range(len(counter)):
    if counter[j] >= MOST_USED_THRESHOLD * max(counter): 
        max_images.append(x_axis[j])

print("Images that are not used = ", zero_images)
print("Images that are used a lot = ", max_images)
print("Total amount of different images = ", MAX_IM - len(zero_images))

# Save the new image and add indicator of total error
tot_error = int(np.sum(current_error.flatten()))
save_name = save_dir + file_name + "_collage_AmountOfImages_" + str(NUMBER_PIXPICS) + "DifferentImages_" + str(MAX_IM - len(zero_images)) + "Total_Error_" + str(tot_error) + ".jpg"
plt.imsave(save_name,reconstructed)

print("Total errror = ",tot_error)

# Visualise the current error and the amount of times each image is used

plt.figure()
plt.imshow(np.array(current_error, dtype=float))
plt.colorbar()

plt.figure()
plt.plot(x_axis, counter, 'o')
plt.show()

###
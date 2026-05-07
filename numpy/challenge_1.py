"""
In computer vision, images are treated as tensors or -3D arrays 
with 3 dimensions(width, height, channels)
Example - A 28 x 28 pixel image of a handwritten digit (like those in MNIST)
is represented as a tensor of shape (28, 28, 1) 
Sometimes sensors have noise or dead pixels which can affect the image


"""

# Create a mock "Grayscale Image" as a 5 x 5 matrix with random integers between 0 and 255.
import numpy as np 
np.random.seed(42)
image = np.random.randint(0,255,(5,5))
print(f"Image =\n{image}\n")

#Identify Noise: Create a boolean mask for all pixels with a value below 50 (too dark) or above 200 (too bright).
noise_mask  = (image < 50) | (image > 200)
print(f"Noise Mask = \n{noise_mask}\n")

mean_image = np.mean(image)
print(f"Mean of image: {mean_image}\n")
# Using that mask, replace all "noisy" pixels with the mean value of the entire image.
noisy_pixels_removed = np.where(noise_mask,mean_image, image)
print(f"Image with noisy pixels removed: \n{noisy_pixels_removed}\n")

# Contrast Boost: Multiply the entire resulting image by 1.2, but ensure no value exceeds 255 (the maximum for an 8-bit image). Hint: Look into np.clip().
constrast_boosted = np.clip(noisy_pixels_removed * 1.2, 0, 255).astype(int)
print(f"Image with contrast boosted: \n{constrast_boosted}\n")
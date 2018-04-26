"""
Splice face images generated by a GAN (specificially Progressive GAN, but not restricted to it) into a face image patch in another image.
The GAN generated image is considered hereof as donor and the image into which the GAN face is spliced is hereof the recipient image.
It is assumed that the donor image has one and only one face, which is typically the case in progressive GAN generated images at the time of writing this code. If this assumption is falsified by modification in the GAN, then it is assumed that this code will become unstable and it is up to the user to modify the code accordingly.

--
MIT License
Copyright (C) 2018 Ashish Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

author: Ashish Gupta
email: ashishagupta@gmail.com
version: 0.1.0

"""

from __future__ import print_function

# import inbuilt libraries
import os
import sys
import argparse

# import third-party libraries
import cv2
import dlib
import numpy

# path to pretrained model utilized in face detection 
# assumes the model file is in the same directory as this code, please specify full path in case this condition is invalid on user's local machine
PREDICTOR_PATH = "pretrained_model.dat"

# pre-defined parameters used by the dlib library towards face detection
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the face patch in donor and recipient images
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the donor image to overlay on the recipient image
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# blur to use during color correction
COLOR_CORRECT_BLUR_FRAC = 0.6

# instantiate objects from dlib library classes for face detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        print('Image seems to have more than one detecable faces present.')
    if len(rects) == 0:
        raise NoFaces

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.
    
    Solve the procrustes problem by subtracting centroids, scaling by the standard deviation, and then using the SVD to calculate the rotation.

    """

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

	"""
    The R we seek is in fact the transpose of the one given by U * Vt. This is because the above formulation assumes the matrix goes on the right
    (with row vectors) where as our solution requires the matrix to be on the left (with column vectors).
    """
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
	"""
	Read the computed facial landmarks
	"""
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return im, s


def warp_im(im, M, dshape):
	"""
	Affine transform donor face image patch to overlay on the recipient face patch with minimal distortion
	"""
	
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colors(im1, im2, landmarks1):
	"""
	Towards perceptual satisfaction of splicing donor image into recipient.
	The color of images is argubly the strongest perceptual attribute, which is ameliorated here.
	Note: Further work will improve on geometric distortion (cylindrical, spherical, etc.) and image intrinsics
	"""
	
    blur_amount = COLOR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Note: to avoid divide-by-zero errors
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))


def splice_donor_recipient(image1, image2, imageout):
	"""
	Splice the donor face patch into the recipient face patch
	"""
	
	im1, landmarks1 = read_im_and_landmarks(image1)
	im2, landmarks2 = read_im_and_landmarks(image2)

	M = transformation_from_points(landmarks1[ALIGN_POINTS],
								   landmarks2[ALIGN_POINTS])

	mask = get_face_mask(im2, landmarks2)
	warped_mask = warp_im(mask, M, im1.shape)
	combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
							  axis=0)

	warped_im2 = warp_im(im2, M, im1.shape)
	warped_corrected_im2 = correct_colors(im1, warped_im2, landmarks1)

	output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
	
	# save the spliced GAN-MediFor image to file
	cv2.imwrite(imageout, output_im)


def process_images():
	"""
	The program reads images in a donor and recipient directories and saves output into another spliced directory.
	By default, the program will assume the directories are:
	donor_directory = './GAN_Faces/'
	recipient_directory = './MediFor_Images/'
	out_directory = './GAN_MediFor/'
	"""
	
	parser = argparse.ArgumentParser(description="Splice image patch for face from GAN generated donor to detected face in recipient image.")
	parser.add_argument("-d", "--donor", dest="donor", default="./GAN_Faces", help="path to directory containing GAN generated faces")
	parser.add_argument("-r", "--recipient", dest="recipient", default="./MediFor_Images", help="path to directory containing images into which faces are spliced")
	parser.add_argument("-o", "--output", dest="output", default="./GAN_MediFor", help="output directory into which spliced images are saved")
	
	args = parser.parse_args()
	donor_directory = args.donor
	recipient_directory = args.recipient
	out_directory = args.output
	
	# donor images
	try:
		head_image_paths = os.listdir(donor_directory)
	except:
		print('Did you create the donor image directory?')
		print('Quiting ...')
		return
		
	# recipient images
	try:
		recipient_paths = os.listdir(recipient_directory)
	except:
		print('Did you create the recipient image directory?')
		print('Quiting ...')
		return
	
	# output folder existence
	if not os.path.exists(out_directory):
		print('Did you create the output image directory?')
		print('Quiting...')
		return
	
	# log errors
	lf = open('./log.txt', 'w')
	
	"""
	Towards the objectives of the MediFor program, all Progressive GAN generated face images are utilized in combination with all available images in recipient images.
	
	Naming convention:
	The spliced images are named as <donor image name>--<recipient image name>.png
	The spliced images can be renamed at a later date if a hashing function is used to rename donor or recipient image file names.	
	"""
	
	
	for head_img in head_image_paths:
		head_path = donor_directory + head_img
		for recipient_img in recipient_paths:
			recipient_path = recipient_directory + recipient_img
			out_img = head_img.split('.')[0] + '--' + recipient_img.split('.')[0] + '.png'
			out_path = out_directory + out_img
			try:
				splice_donor_recipient(recipient_path, head_path, out_path)
				print('{}, {}, {}'.format(head_path, recipient_path, out_path))
			except Exception as err:
				print(err)
				lf.write('Issue with: {}\n'.format(out_img))
	
	lf.close()		

if __name__ == '__main__':
	"""
	Please read the documentation to set the data in appropriate directories.
	The program will read images from these directories
	"""
	process_images()
	





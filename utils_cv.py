import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.transform import ProjectiveTransform, warp
from skimage import io
from skimage.feature import canny, match_template
from skimage.transform import rescale
from skimage.morphology import dilation, disk
import cv2
import pickle

def save_pkl(variable, name):
    name = name + '.pkl'
    output = open(name, 'wb')
    pickle.dump(variable, output)
    output.close()

def load_pkl(name):
    name = name + '.pkl'
    pkl_file = open(name, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result

def show(image):
    plt.figure(figsize = (7, 7), dpi = 100)
    plt.imshow(image, cmap = 'gray')
    plt.show()
    
def read_images(array_of_paths):
    array_of_images = []
    for img_path in array_of_paths:
        image = cv2.imread(img_path)
        array_of_images.append(image)
    return array_of_images
    
def print_images(array_of_images, nrows, ncols):
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (20, 20))
    for ax, image in zip(axes.flat, array_of_images):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
    plt.tight_layout()
    plt.show()


def divide_into_boxes(image):
    box_size = int(image.shape[0]/9)
    return [cv2.resize(image[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size], (100, 100)) for i in range(9) for j in range(9)]

def adjusting_brightness(image, a = 1.1, b = 0):
    return cv2.convertScaleAbs(image, alpha=a, beta=b)

# def matrix(boxes, templates, correlation_threshold, alpha, beta):
#     result = np.array([], dtype = 'int8')

#     for box in boxes:
#         box = adjusting_brightness(box, alpha, beta)
#         number = 0
#         correlations = []
#         corr_res = [match_template(box, templates[i], pad_input = True) for i in range(18)]
#         correlations = [i.max() for i in corr_res]
#         if max(correlations) > correlation_threshold:
#             number = int(correlations.index(max(correlations))) + 1
#             if int(number) > 9:
#                 number = int(number - 9)
#         result = np.append(result, number)
#     return result.reshape((9,9))

# def recognize_digits(image, templates, correlation_threshold = 0.3, alpha = 1.1, beta = 0):
#     boxes = divide_into_boxes(image)
#     templates = [cv2.resize(i, (90, 90)) for i in templates]
#     return matrix(boxes, templates, correlation_threshold, alpha, beta)


# def plot_results(image, result, solution, alpha, beta):
#     n, m = image.shape
#     padding = n/9

#     x_coordinates = np.array(range(0, 9))*(padding) + padding*0.4
#     y_coordinates = np.array(range(0, 9))*(padding) + padding*0.7

#     plt.figure(figsize = (5, 5))
#     plt.imshow(adjusting_brightness(image, alpha, beta), cmap = 'gray')
#     for i, x in enumerate(x_coordinates):
#         for j, y in enumerate(y_coordinates):
#             if result[j, i] == 0:
#                 plt.text(x, y, solution[j, i], fontsize = 20, c = 'c')
#             else:
#                 plt.text(x + padding*0.2, y + padding*0.2, solution[j, i], fontsize = 10, c = 'r')

#     plt.show()
    
# def whole_pipeline(image, templates, alpha = 1.1, beta = 0, correlation = 0.3, plot = True, printing = True):
#     image = normalize_image(image)
#     result = recognize_digits(image, templates, correlation, alpha, beta)
#     if printing:
#         print(result)
#     try:
#         solution = solve_sudoku(result)
#         if printing:
#             print('\n')
#             print(solution)
#         if plot:
#             plot_results(image, result, solution, alpha, beta)
#     except:
#         print('Something went wrong :(, the original was:')
#         show(image)


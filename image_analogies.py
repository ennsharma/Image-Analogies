import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.transform import rescale
from sklearn.neighbors import LSHForest
from skimage.color import rgb2yiq, yiq2rgb
import os, errno
import nmslib

# Hyperparameters
L = 1
K = 0.75

# Kernels
GAUSSIAN_SMALL = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]
                 ]) / 16

GAUSSIAN_LARGE = np.array([[1, 4,  6,  4,  1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4,  6,  4,  1]
                 ]) / 256

# File Constants
INPUT = './input/'
OUTPUT = './output/'

A_NAME = 'A.jpg'
A_PRIME_NAME = 'B.jpg'
B_NAME = 'A_prime.jpg'
B_PRIME_NAME = 'B_prime.jpg'

USE_LUMINANCE = True

# Feature Vector Storage
vector_cache = {}
lshf_cache = {}

# Feature Extraction
def get_features_and_offsets(i, j, offset_size, l, pyramid, kernel):
    features, offsets = [], []
    for i_offset in range(-offset_size, offset_size+1):
        for j_offset in range(-offset_size, offset_size+1):
            i_prime, j_prime = i + i_offset, j + j_offset
            if i_prime >= 0 and i_prime < pyramid[l].shape[0] and j_prime >= 0 and j_prime < pyramid[l].shape[1] and np.sum(pyramid[l][i_prime][j_prime]):
                weight = kernel[i_offset+offset_size][j_offset+offset_size]
                features.append((pyramid[l][i_prime][j_prime], weight))
                offsets.append((i_offset, j_offset, weight))
    return features, offsets

def get_features_only(i, j, offsets, l, pyramid):
    features = []
    for i_offset, j_offset, weight in offsets:
        i_prime, j_prime = i + i_offset, j + j_offset
        if i_prime >= 0 and i_prime < pyramid[l].shape[0] and j_prime >= 0 and j_prime < pyramid[l].shape[1]:
            features.append((pyramid[l][i_prime][j_prime], weight))
        else:
            features.append((np.array([0, 0, 0]), 0))
    return features

def construct_F(i, j, l, pyramid, pyramid_prime, offsets=None, is_A=True):
    if offsets:
        size = sum([len(x) for x in offsets])
        if (i, j, l, is_A, size) in vector_cache:
            return vector_cache[(i, j, l, is_A, size)]
        high_res_bp_features = get_features_only(i, j, offsets[0], l, pyramid_prime)
        high_res_b_features = get_features_only(i, j, offsets[1], l, pyramid)
        if l+1 != L:
            low_res_bp_features = get_features_only(i, j, offsets[2], l-1, pyramid_prime)
            low_res_b_features = get_features_only(i, j, offsets[3], l-1, pyramid_prime)
            features = high_res_bp_features + high_res_b_features + low_res_bp_features + low_res_b_features
        else:
            features = high_res_bp_features + high_res_b_features
        vector_cache[(i, j, l, is_A, size)] = features
        return features
    else:
        high_res_bp_features, high_res_bp_offsets = get_features_and_offsets(i, j, 2, l, pyramid_prime, GAUSSIAN_LARGE)
        high_res_b_features, high_res_b_offsets = get_features_and_offsets(i, j, 2, l, pyramid, GAUSSIAN_LARGE)
        if l+1 != L:
            low_res_bp_features, low_res_bp_offsets = get_features_and_offsets(i//2, j//2, 1, l-1, pyramid_prime, GAUSSIAN_SMALL)
            low_res_b_features, low_res_b_offsets = get_features_and_offsets(i//2, j//2, 1, l-1, pyramid, GAUSSIAN_SMALL)
            features = high_res_bp_features + high_res_b_features + low_res_bp_features + low_res_b_features
            offsets = [high_res_bp_offsets, high_res_b_offsets, low_res_bp_offsets, low_res_b_offsets]
        else:
            features = high_res_bp_features + high_res_b_features
            offsets = [high_res_bp_offsets, high_res_b_offsets]
        size = sum([len(x) for x in offsets])
        vector_cache[(i, j, l, is_A, size)] = features
        return features, offsets

# Construction Methods
def construct_pyramid(im):
    pyramid = [im]
    for i in range(L-1):
        pyramid.append(rescale(image=pyramid[i], scale=0.5, mode='reflect'))
    return pyramid

def construct_normalized_vector(F, feature_size=3):
    feature_vector, weights, i = np.zeros((len(F)*3,)), [], 0
    for feature, weight in F:
        for j in range(feature_size):
            feature_vector[i+j] = feature[j]
        weights.append(weight)
        i = i + feature_size
    return feature_vector / np.linalg.norm(feature_vector), weights

def construct_lshf(pyramid, prime_pyramid, l, query, offsets):
    if (l, query.shape) in lshf_cache:
        return lshf_cache[(l, query.shape)]

    X_train, idx_map, idx = [], {}, 0
    for i_a in range(prime_pyramid[l].shape[0]):
        for j_a in range(prime_pyramid[l].shape[1]):
            features = construct_F(i_a, j_a, l, pyramid, prime_pyramid, offsets=offsets)
            neighbor = construct_normalized_vector(features)[0]
            if neighbor.shape == query.shape:
                X_train.append(neighbor)
                idx_map[idx] = (i_a, j_a)
                idx += 1
    X_train = np.array(X_train)

    # Construct NN model
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(X_train)
    index.createIndex({'post': 2}, print_progress=True)
    lshf_cache[(l, query.shape)] = (index, X_train, idx_map)

    return index, X_train, idx_map

def compute_weighted_difference(normalized_1, weights_1, normalized_2, weights_2):
    difference = np.zeros(normalized_1.shape)
    for i in range(normalized_1.shape[0]):
        difference[i] = normalized_1[i] * weights_1[i//3] - normalized_2[i] * weights_2[i//3]
    return difference

def compute_distance(F_1, F_2):
    normalized_1, weights_1 = construct_normalized_vector(F_1)
    normalized_2, weights_2 = construct_normalized_vector(F_2)
    difference = compute_weighted_difference(normalized_1, weights_1, normalized_2, weights_2)
    return np.linalg.norm(difference)

def compute_luminance_transforms(A, B):
    lum_A, lum_B = A[:,:,0], B[:,:,0]
    mu_A, std_A = np.mean(lum_A), np.std(lum_A)
    mu_B, std_B = np.mean(lum_B), np.std(lum_B)
    return np.vectorize(lambda x: (std_A/std_B)*(x - mu_A) + mu_B), np.vectorize(lambda x: (std_B/std_A)(x - mu_B) + mu_A)

# Texture Synthesis
def best_approximate_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i_b, j_b):
    # Construct feature vectors
    features, offsets = construct_F(i_b, j_b, l, B_pyramid, B_prime_pyramid, is_A=False)
    query = construct_normalized_vector(features)[0]
    lshf, X_train, idx_map = construct_lshf(A_pyramid, A_prime_pyramid, l, query, offsets)
    ids, indices = lshf.knnQuery(query, k=1)
    return idx_map[ids[0]]

def best_coherence_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i, j):
    best_i_prime, best_j_prime, min_norm = None, None, float("inf")
    for i_offset in range(-2, 3):
        for j_offset in range(-2, 3):
            i_prime, j_prime = i + i_offset, j + j_offset
            if i_prime >= 0 and i_prime < B_prime_pyramid[l].shape[0] and j_prime >= 0 and j_prime < B_prime_pyramid[l].shape[1] and (i_prime, j_prime) in s:
                x, y = i - i_prime + s[(i_prime, j_prime)][0], j - j_prime + s[(i_prime, j_prime)][1]
                if x >= 0 and x < B_prime_pyramid[l].shape[0] and y >=0 and y < B_prime_pyramid[l].shape[1]:
                    F_1, offsets = construct_F(x, y, l, A_pyramid, A_prime_pyramid)
                    F_2 = construct_F(i, j, l, B_pyramid, B_prime_pyramid, offsets=offsets, is_A=False)
                    norm = compute_distance(F_1, F_2)
                    if norm < min_norm:
                        best_i_prime, best_j_prime, min_norm = i_prime, j_prime, norm
    if (best_i_prime, best_j_prime) == (None, None):
        return (best_i_prime, best_j_prime)
    return i - best_i_prime + s[(best_i_prime, best_j_prime)][0], j - best_j_prime + s[(best_i_prime, best_j_prime)][1]

def best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i_b, j_b):
    # Compute matches
    i_a_app, j_a_app = best_approximate_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i_b, j_b)
    i_a_coh, j_a_coh = best_coherence_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i_b, j_b)
    if (i_a_coh, j_a_coh) == (None, None):
        return i_a_app, j_a_app

    # Compute neighborhood feature vector distances
    F_q, offsets = construct_F(i_b, j_b, l, B_pyramid, B_prime_pyramid, is_A=False)
    F_app = construct_F(i_a_app, j_a_app, l, A_pyramid, A_prime_pyramid, offsets=offsets)
    F_coh = construct_F(i_a_coh, j_a_coh, l, A_pyramid, A_prime_pyramid, offsets=offsets)

    d_app = compute_distance(F_app, F_q)
    d_coh = compute_distance(F_coh, F_q)

    if d_coh > d_app * (1 + 2**(L - l) * K):
        return i_a_app, j_a_app
    return i_a_coh, j_a_coh

def create_image_analogy(A, A_prime, B):
    # Compute Gaussian pyramids for A, A', and B
    A_pyramid = construct_pyramid(A)
    A_prime_pyramid = construct_pyramid(A_prime)
    B_pyramid = construct_pyramid(B)
    B_prime_pyramid = [np.zeros(B_pyramid[i].shape) for i in range(len(A_prime_pyramid))]

    # Compute the best match for each pixel
    s = {}
    for l in range(L-1, -1, -1):
        print("Processing Level %d." %(L-l))
        for i_b in range(B_prime_pyramid[l].shape[0]):
            print("\tRow %d" %i_b)
            for j_b in range(B_prime_pyramid[l].shape[1]):
                i_a, j_a = best_match(A_pyramid, A_prime_pyramid, B_pyramid, B_prime_pyramid, s, l, i_b, j_b)
                B_prime_pyramid[l][i_b][j_b] = A_prime_pyramid[l][i_a][j_a]
                s[(i_b, j_b)] = (i_a, j_a)
    return B_prime_pyramid[0]

if __name__ == '__main__':
    A = plt.imread(INPUT + A_NAME)
    A_prime = plt.imread(INPUT + A_PRIME_NAME)
    B = plt.imread(INPUT + B_NAME)
    if USE_LUMINANCE:
        A, A_prime, B = rgb2yiq(A), rgb2yiq(A_prime), rgb2yiq(B)
        transform_func, inverse_transform_func = compute_luminance_transforms(A, B)
        A[:,:,0] = transform_func(A[:,:,0])
        A_prime[:,:,0] = transform_func(A_prime[:,:,0])
        B[:,:,0] = transform_func(B[:,:,0])

    B_prime = create_image_analogy(A, A_prime, B)
    if USE_LUMINANCE:
        B_prime[:,:,0] = inverse_transform_func(B_prime[:,:,0])
        B_prime = yiq2rgb(B_prime)

    try:
        os.makedirs(OUTPUT)
    except:
        pass
    plt.imsave(OUTPUT + B_PRIME_NAME, B_prime/255.)

import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def svm_train_brute(training_data):
    # convert data to np array just in case
    training_data = np.asarray(training_data)

    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]

    # initial margin with negative value
    margin = -9999999
    # we will use new variables because they will update
    s_last, w_last, b_last = None, None, None

    # in 2D data we need only 2 or 3 support vectors
    # we will look every couple labels for finding the max margin

    # one positive - one negative sv
    for pos in positive:
        for neg in negative:
            mid_point = (pos[0:2] + neg[0:2]) / 2
            w = np.array(pos[:-1] - neg[:-1])
            w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
            b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

            if margin <= compute_margin(training_data, w, b):
                margin = compute_margin(training_data, w, b)
                s_last = np.array([pos, neg])
                w_last = w
                b_last = b
    # two positive - one negative sv
    for pos in positive:
        for pos1 in positive:
            for neg in negative:
                if (pos[0] != pos1[0]) and (pos[1] != pos1[1]):
                    separator = (pos1 - pos)[:2]
                    ws = separator / np.sqrt(separator.dot(separator))
                    # projected point
                    # projection = neg[:2] + ws.dot((neg - pos)[:2]) * ws
                    projection = np.append(pos[:2] + (np.dot(ws, (neg[:2] - pos[:2]))) * ws, [1])

                    # we use projected point to find mid point
                    mid_point = (projection[0:2] + neg[0:2]) / 2
                    w = np.array(projection[:-1] - neg[:-1])
                    # w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
                    w = w / np.sqrt(w.dot(w))
                    b = -1 * (w.dot(mid_point))

                    if margin <= compute_margin(training_data, w, b):
                        margin = compute_margin(training_data, w, b)
                        s_last = np.array([pos, pos1, neg])
                        w_last = w
                        b_last = b

    # one positive - two negative sv
    for neg in negative:
        for neg1 in negative:
            for pos in positive:
                if neg[0] != neg1[0] and neg[1] != neg1[1]:
                    separator = (neg1[:2] - neg[:2])
                    ws = separator / np.sqrt(separator.dot(separator))
                    # projected point
                    projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [-1])

                    # we use projected point to find mid point
                    mid_point = (pos[0:2] + projection[0:2]) / 2
                    w = np.array(pos[:-1] - projection[:-1])
                    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
                    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

                    if margin <= compute_margin(training_data, w, b):
                        margin = compute_margin(training_data, w, b)
                        s_last = np.array([pos, neg, neg1])
                        w_last = w
                        b_last = b
    return w_last, b_last, s_last


# distance can't be negative we return absolute value
def distance_point_to_hyperplane(pt, w, b):
    return np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1]))))


def compute_margin(data, w, b):
    margin = distance_point_to_hyperplane(data[0, :-1], w, b)

    for pt in data:
        distance = distance_point_to_hyperplane(pt[:-1], w, b)
        if distance < margin:
            margin = distance_point_to_hyperplane(pt[:-1], w, b)
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0

    return margin


def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1


def svm_train_multiclass(training_data):
    w2, b2 = [], []
    training_data = np.array(training_data)
    num_of_classes = training_data[1]

    # we don't have class labels with 0, so we need to start from 1
    for y in range(1, num_of_classes + 1):
        only_data = np.copy(training_data[0])
        for dt in only_data:
            # one vs rest
            if dt[2] == y:
                dt[2] = 1
            else:
                dt[2] = -1
        wtmp, btmp, stmp = svm_train_brute(only_data)

        w2.append(wtmp)
        b2.append(btmp)

    return [w2, b2]


def svm_test_multiclass(W, B, x):
    # initial label
    label = -1
    # initial distance from data point to hyperplane
    dist_from_hyper = 0
    for i in range(0, len(W)):
        # we are predicting the current label as one vs rest
        pred = svm_test_brute(W[i], B[i], x)
        # distance to hyperplane
        tmp_dist = np.abs(distance_point_to_hyperplane(x, W[i], B[i]))

        # if prediction is correct
        # and just in case we control the distances
        if pred == 1 and tmp_dist > dist_from_hyper:
            label = i
            dist_from_hyper = tmp_dist

    # iteration starts from 0 due to the array restricts, we add 1 to return correct label
    return label + 1


def plot_hyper_binary(w, b, data):
    line = np.linspace(-100, 100)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line - b) / w[1])
    else:
        plt.axvline(x=b)
    plot_training_data_binary(data)


def plot_multi(data, W, B):
    x = np.linspace(-10, 10, 1000)
    for c in range(len(W)):
        if W[c][0] == 0:
            plt.axhline(-B[c] / W[c][1])
        elif W[c][1] == 0:
            plt.axvline(x=-B[c] / W[c][0])
        else:
            m = -W[c][0] / W[c][1]
            yint = -B[c] / W[c][1]
            plt.plot(x, m * x + yint)

    plot_training_data_multi(data)
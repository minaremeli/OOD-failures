# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cvxpy as cp
import torch


def fit_max_margin(x_train, y_train, verbose=False):
    """
    Returns max-margin solution on the training datapoint
    # Arguments:
        x_train (np.array): training inputs
        y_train (np.array): 0/1 training labels
    # Returns
        weights (np.array): array of weights
        bias (float): bias value
    """

    # One could also use scipy to do this but
    # with the following code there's greater flexibility to play around with the
    # constraints and see how things change

    x_train = x_train.reshape((x_train.shape[0], -1))
    y_train = y_train.reshape((y_train.shape[0]))
    A = np.hstack([x_train, np.ones((x_train.shape[0], 1))])  # Append a "1" for the bias feature
    Ide = np.identity(x_train.shape[1])
    b_ones = np.ones(A.shape[0])
    cp_weights = cp.Variable(A.shape[1])

    # Quadratic program corresponding to minimizing ||w||^2
    # subject to y (x^T w) >= 1
    # prob = cp.Problem(cp.Minimize(cp.quad_form(cp_weights[:-1], Ide)),
    prob=cp.Problem(cp.Minimize(cp.norm(cp_weights[:-1])), # rewrite problem using norms: http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms
                                      [np.diag(2 * y_train - 1) @ ((A @ cp_weights)) >= b_ones])
    prob.solve(verbose=verbose, solver=cp.SCS)
    if prob.status == 'infeasible':
        print("The problem is infeasible!")
        return None, None
    else:
        weights = cp_weights.value
        return weights[:-1], weights[-1]


def evaluate_max_margin(x_test, y_test, weights, bias):
    """
    Returns accuracy of a linear classifier on test data
    # Arguments:
        x_test (np.array): test inputs
        y_test (np.array): 0/1 test labels
        weights, bias (np.array, float): weights and bias
    # Returns
        accuracy: accuracy of the weights on the given test data
    """
    x_test = x_test.reshape((x_test.shape[0], -1))
    y_test = y_test.reshape((y_test.shape[0]))
    margins = np.matmul(x_test, weights) + bias
    print(margins.shape)
    print(y_test.shape)
    accuracy = torch.mean((torch.multiply(margins, 2 * y_test - 1) > 0.0).float())
    return accuracy


def l2_norm(weights):
    return np.linalg.norm(weights)
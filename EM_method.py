"""
Probabilistic robotics HW
Implementation "learn intrinsic parameter" pseudocode in Probabilistic robotics
Author : Taeyoung Kim (tyoung96@yonsei.ac.kr)
"""

import numpy as np


# Gaussian Function
def gaussian(x, mean, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- (x - mean) ** 2 / (2 * sigma ** 2))


"""
Load sensor data
inputData[0,:] - z (true range)
inputData[1,:] - output
"""
inputData = np.load('sensors.npz')['D']

sensor_max = 50.0  # Sensor max range
data_num = 3000

# Initial guesses for intrinsic parameter
_mix_density = np.random.dirichlet(np.ones(4), size=1)  # shape (1,4)
z_hit = _mix_density[:, 0].item()
z_short = _mix_density[:, 1].item()
z_max = _mix_density[:, 2].item()
z_rand = _mix_density[:, 3].item()

_sigma = float(np.random.randint(1, 10))
_lamb_short = np.random.rand(1).item()

convergence = False  # Check for convergence

while True:
    # Previous value
    prev = np.array([z_hit, z_short, z_max, z_rand, _sigma, _lamb_short], dtype=float)

    # Sum of e_i
    e_hit_sum = 0.0
    e_short_sum = 0.0
    e_max_sum = 0.0
    e_rand_sum = 0.0

    e_short_list = np.array([])  # For calculate _lamb_short
    _cal_sigma_tmp = 0.0  # For calculate _sigma
    _cal_sigma_tmp_sum = 0.0

    for i in range(data_num):
        z = inputData[0, i]  # true range (in pseudo code : z_i^*)
        out = inputData[1, i]  # output in `sensor_data.py` (in pseudo code : z_i)

        # Calculate hit mode probability
        normalize_hit = 0.0
        for j in range(int(sensor_max)):
            normalize_hit += gaussian(j, z, _sigma)
        normalize_hit = 1. / normalize_hit

        p_hit = gaussian(out, z, _sigma) * normalize_hit

        # Calculate short mode probability
        if out <= z:
            normalize_short = 1 - np.exp(-1 * _lamb_short * z)
            p_short = _lamb_short * np.exp(-1 * _lamb_short * out) / normalize_short
        else:
            p_short = 0.0

        # Calculate max mode probability
        if out == sensor_max:
            p_max = 1.0
        else:
            p_max = 0.0

        # Calculate rand mode probability
        p_rand = 1.0 / sensor_max

        p = np.array([np.float_(p_hit), np.float_(p_short), p_max, p_rand])

        p_z = np.dot(_mix_density, p)  # (p_hit*z_hit) + (p_short*z_short) + (p_max*z_max) + (p_rand*z_rand)

        # Expectation
        e_hit = np.float_(p_hit * z_hit / p_z)
        e_short = np.float_(p_short * z_short / p_z)
        e_max = np.float_(p_max * z_max / p_z)
        e_rand = np.float_(p_rand * z_rand / p_z)

        e_short_list = np.append(e_short_list, e_short)
        _cal_sigma_tmp = e_hit * (out - z) ** 2
        _cal_sigma_tmp_sum += _cal_sigma_tmp

        # Sum of expectation
        e_hit_sum += e_hit
        e_short_sum += e_short
        e_max_sum += e_max
        e_rand_sum += e_rand

    # Maximization
    z_hit = e_hit_sum / float(data_num)
    z_short = e_short_sum / float(data_num)
    z_max = e_max_sum / float(data_num)
    z_rand = e_rand_sum / float(data_num)
    _mix_density = [z_hit, z_short, z_max, z_rand]
    _sigma = (_cal_sigma_tmp_sum / e_short_sum) ** (1 / 2)
    _lamb_short = np.float_(e_short_sum / (np.matmul(e_short_list.reshape(1, -1), inputData[1, :])))

    # Current value
    cur = np.array([z_hit, z_short, z_max, z_rand, _sigma, _lamb_short], dtype=float)

    # Check for convergence
    convergence = np.allclose(prev, cur, atol=1e-01)
    if convergence:
        break

# Result
print("z_hit : ", z_hit)
print("z_short : ", z_short)
print("z_max : ", z_max)
print("z_rand : ", z_rand)
print("_sigma : ", _sigma)
print("_lamb_short : ", _lamb_short)

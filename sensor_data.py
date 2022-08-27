import numpy as np
import matplotlib.pyplot as plt


z_max = 50  # Sensor range
mix_density = [0.7, 0.2, 0.05, 0.05]  #Mixture density [z_hit, z_short, z_max, z_rand]
sigma = np.sqrt(5)  # standard deviation of Gaussian (hit)
lamb_short = 0.3      # lambda of exponential dist (short)

data_num = 3000
outputs = np.zeros((2,data_num))

for i in range(data_num):
    #mode selection
    mode = np.random.choice(len(mix_density), 1, p=list(mix_density)).item()

    z = np.random.uniform(0, z_max, size=1).item()  # Current distance
    # z = 20 # 실제 센서에서 나와야 하는 관찰 값

    if mode == 0:       # hit mode
        while True:
            out = np.random.normal(z, sigma, 1).item()
            if (out >= 0) and (out <=z_max):
                break
        outputs[:,i] = z, out
    elif mode == 1:     #Short mode
        while True:
            out = np.random.exponential(scale = 1/lamb_short, size =1).item()
            if (out >= 0 ) and (out <=z):
                break
        outputs[:,i] = z, out
    elif mode == 2:     # Max
        outputs[:,i] = z, z_max
    else:              #Rand
        outputs[:,i] = z, np.random.uniform(0,z_max,size = 1).item()


plt.plot(outputs[1,:], 'o')
plt.show()
np.savez('./sensors.npz', D=outputs)
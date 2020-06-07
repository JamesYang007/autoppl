import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5
])

# AutoPPL benchmark in seconds
autoppl_res = np.array([
    0.184537766,
    0.792546206,
    1.273571554,
    3.687687061,
    6.339291371,
    16.626929522,
    23.583710674
])

# STAN benchmark in seconds
stan_res = np.array([
    0.520629,
    0.698193,
    2.45387,
    9.30525,
    13.3364,
    35.1523,
    53.9648
])

plt.plot(n_range, autoppl_res, '-',
         marker='o', color='blue',
         label='autoppl', alpha=0.5)
plt.plot(n_range, stan_res, '-',
         marker='o', color='red',
         label='stan', alpha=0.5)
plt.title('Regression Benchmark')
plt.xlabel('Number of Samples Drawn')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('runtime.png')
plt.show()

width=0.3
eps=0.15
plt.bar([-width/2-eps, width/2+eps], [4.62, 9.98808],
        width=0.3,
        color=['blue', 'red'],
        tick_label=['autoppl', 'stan'],
        alpha=0.5)
plt.title('Compilation Time')
plt.ylabel('Time (s)')
plt.savefig('compiletime.png')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5
])

# AutoPPL warmup in seconds
autoppl_warmup = np.array([
    0.22556,
    0.497538,
    1.11476,
    2.94345,
    4.33,
    10.5489,
    15.0221
])

autoppl_sampling = np.array([
    0.129868,
    0.648974,
    0.925075,
    1.49762,
    2.5439,
    8.52077,
    7.91238
])

autoppl_total = autoppl_warmup + autoppl_sampling

autoppl_esss = np.array([
   [30.5258,
   10.0393,
   10.6004,
   26.5849,
   10.9389],

   [5.1383e+02,
   4.5337e+02,
   6.6654e+02,
   4.3830e+02,
   3.6803e+02],

   [4.1271e+02,
   5.8306e+02,
   8.8573e+02,
   4.1978e+02,
   6.7982e+02],

   [1.7385e+03,
   1.6557e+03,
   2.6217e+03,
   1.4613e+03,
   1.6672e+03],

   [1.5615e+03,
   2.0537e+03,
   3.5358e+03,
   1.7204e+03,
   2.1638e+03],

   [2.6499e+03,
   3.3248e+03,
   5.1976e+03,
   2.5220e+03,
   3.1148e+03],

   [6.9521e+03,
   7.9424e+03,
   1.2444e+04,
   6.7155e+03,
   7.8394e+03]
])

# STAN warmup in seconds
stan_warmup = np.array([
    0.17438,
    0.844057,
    2.15675,
    7.00261,
    9.73703,
    25.5543,
    37.6729
])

stan_sampling = np.array([
    0.729614,
    1.40904,
    2.04203,
    3.63673,
    6.3536,
    11.977,
    21.6173
])

stan_total = stan_warmup + stan_sampling

stan_esss = np.array([

    [4.30656,
    4.27916,
    11.63570,
    4.57355,
    12.96620],

    [105.885,
    152.669,
    298.816,
    125.940,
    218.573],

    [245.855,
    304.226,
    478.066,
    242.390,
    278.914],

    [648.843,
     689.463,
    1153.790,
     562.155,
     631.185],

    [730.396,
     922.350,
    1415.310,
     651.911,
     835.103],

    [2290.37,
    2838.39,
    4264.28,
    2198.88,
    2739.03],

    [2633.79,
    2947.76,
    4620.26,
    2488.19,
    2880.71]
])

colors = ['blue', 'red', 'green', 'orange', 'cyan']

# Plot warmup and sampling times
plt.plot(n_range, autoppl_warmup, '-',
         marker='o', color=colors[0],
         label='autoppl warmup', alpha=0.5)
plt.plot(n_range, autoppl_sampling, '-',
         marker='o', color=colors[1],
         label='autoppl sampling', alpha=0.5)
plt.plot(n_range, autoppl_total, '-',
         marker='o', color=colors[2],
         label='autoppl total', alpha=0.5)

plt.plot(n_range, stan_warmup, '--',
         marker='o', color=colors[0],
         label='STAN warmup', alpha=0.5)
plt.plot(n_range, stan_sampling, '--',
         marker='o', color=colors[1],
         label='STAN sampling', alpha=0.5)
plt.plot(n_range, stan_total, '--',
         marker='o', color=colors[2],
         label='STAN total', alpha=0.5)

plt.title('Warmup/Sampling Time')
plt.xlabel('Number of Samples Drawn')
plt.ylabel('Time (s)')
plt.legend()
plt.savefig('runtime.png')
plt.show()

# Plot ESS/s
for j in range(autoppl_esss.shape[1]):
    plt.plot(n_range, autoppl_esss[:,j], '-',
             marker='o', color=colors[j],
             label='autoppl p[{j}]'.format(j=j), alpha=0.5)

    plt.plot(n_range, stan_esss[:,j], '--',
             marker='o', color=colors[j],
             label='STAN p[{j}]'.format(j=j), alpha=0.5)

plt.title('ESS per time (s)')
plt.xlabel('Number of Samples Drawn')
plt.ylabel('ESS/s')
plt.legend()
plt.savefig('ess_s.png')
plt.show()

#width=0.3
#eps=0.15
#plt.bar([-width/2-eps, width/2+eps], [4.62, 9.98808],
#        width=0.3,
#        color=['blue', 'red'],
#        tick_label=['autoppl', 'stan'],
#        alpha=0.5)
#plt.title('Compilation Time')
#plt.ylabel('Time (s)')
#plt.savefig('compiletime.png')
#plt.show()

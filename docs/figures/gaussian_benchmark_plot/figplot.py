import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 3e3, 5e3, 7e3, 1e4, 2e4
])


# AutoPPL warmup in seconds
autoppl_warmup = np.array([
    0.00458256,
    0.0730167,
    0.16297,
    0.525782,
    0.920745,
    1.22564,
    1.82999,
    3.63635
])

autoppl_sampling = np.array([
    0.0112714,
    0.105494,
    0.227139,
    0.651312,
    1.07762,
    1.51046,
    2.11578,
    4.11572
])

autoppl_total = autoppl_warmup + autoppl_sampling

autoppl_esss = np.array([

   [2.0302e+03,
   2.0336e+03,
   6.8007e+03],

   [1.3868e+03,
   1.3855e+03,
   1.8477e+03],

   [1.5635e+03,
   1.5639e+03,
   1.7158e+03],

   [1.3485e+03,
   1.3500e+03,
   1.5935e+03],

   [9.9387e+02,
   9.9422e+02,
   1.8011e+03],

   [1.2316e+03,
   1.2318e+03,
   1.5084e+03],

   [1.2571e+03,
   1.2571e+03,
   1.6564e+03],

   [1.3664e+03,
   1.3666e+03,
   1.7869e+03]

])

# STAN warmup in seconds
stan_warmup = np.array([
    0.096904,
    0.869262,
    2.0625,
    6.20375,
    10.7074,
    15.2034,
    21.8447,
    43.8083
])

stan_sampling = np.array([
    0.093209,
    1.13973,
    2.4933,
    7.58344,
    12.8239,
    17.9216,
    24.5716,
    51.418
])

stan_total = stan_warmup + stan_sampling

stan_esss = np.array([

    [161.880,
    161.871,
    1386.460],

    [108.7730,
    108.7380,
    149.9360],

    [96.7849,
     96.7470,
    136.5910],

    [100.930,
    100.890,
    136.955],

    [103.602,
    103.597,
    126.952],

    [95.7787,
    95.7338,
    123.0780],

    [115.262,
    115.239,
    161.244],

    [102.004,
    101.986,
    141.212]

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

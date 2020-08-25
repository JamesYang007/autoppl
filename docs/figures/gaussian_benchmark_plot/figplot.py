import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 3e3, 5e3, 7e3, 1e4, 2e4
])

# AutoPPL warmup in seconds
autoppl_warmup = np.array([
    0.004123,
    0.047543,
    0.104441,
    0.335634,
    0.581569,
    0.829182,
    1.1388,
    2.29912
])

autoppl_sampling = np.array([
    0.005388,
    0.074479,
    0.143482,
    0.40151,
    0.716135,
    0.98969,
    1.3699,
    2.67381
])

autoppl_total = autoppl_warmup + autoppl_sampling

autoppl_esss = np.array([

   [1.1323e+03,
    1.1315e+03,
    1.1210e+04],

   [1.3918e+03,
    1.3961e+03,
    2.7399e+03],

   [1.8553e+03,
    1.8574e+03,
    2.4084e+03],

   [1.7613e+03,
    1.7608e+03,
    1.8466e+03],

   [1.3856e+03,
    1.3860e+03,
    2.2567e+03],

   [2.0913e+03,
    2.0916e+03,
    2.7971e+03],

   [1.5422e+03,
    1.5418e+03,
    2.4278e+03],

   [1.8775e+03,
    1.8773e+03,
    2.4351e+03]

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

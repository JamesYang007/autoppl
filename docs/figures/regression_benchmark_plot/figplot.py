import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5
])

# AutoPPL warmup in seconds
autoppl_warmup = np.array([
    0.0478412,
    0.169495,
    0.37302,
    1.14783,
    1.65571,
    3.82485,
    5.40454
])

autoppl_sampling = np.array([
    0.109465,
    0.337762,
    0.318597,
    0.666965,
    1.23373,
    2.41495,
    2.92585
])

autoppl_total = autoppl_warmup + autoppl_sampling

autoppl_esss = np.array([

    [42.3012,
    52.4322,
    48.8026,
    91.063,
    86.7563],

    [854.616,
    720.024,
    922.783,
    911.901,
    1288.14],

    [1695.29,
    1306.06,
    1230.11,
    1503.71,
    2538.5],

    [3689.11,
    3215.53,
    3250.83,
    4162.65,
    5534.6],

    [4299.1,
    3668.12,
    3740.65,
    4546.69,
    7151.87],

    [11137.1,
    9073.22,
    9946.83,
    10682.4,
    19027.5],

    [20931.7,
    17046,
    17938.3,
    21291.3,
    31679.9]
])

# STAN warmup in seconds
stan_warmup = np.array([
    0.055,
    0.901,
    1.632,
    6.298,
    9.727,
    23.173,
    34.934
])

stan_sampling = np.array([
    0.585,
    1.656,
    1.735,
    4.142,
    5.719,
    13.885,
    18.584
])

stan_total = stan_warmup + stan_sampling

stan_esss = np.array([

    [5.75404,
    9.53699,
    5.80000,
    5.70000,
    164.00000],

    [108.016,
    109.984,
    108.000,
    160.000,
    320.000],

    [380.834,
    159.806,
    188.000,
    324.000,
    436.000],

    [666.723,
    535.699,
    579.000,
    626.000,
    1160.000],

    [867.085,
    793.692,
    864.000,
    928.000,
    1664.000],

    [2015.28,
    1495.92,
    1634.00,
    2002.00,
    3350.00],

    [3150.01,
    2721.23,
    2879.00,
    3151.00,
    5273.00]
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

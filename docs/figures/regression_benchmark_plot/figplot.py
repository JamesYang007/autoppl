import matplotlib.pyplot as plt
import numpy as np

# N values
n_range = np.array([
    1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5
])

# AutoPPL warmup in seconds
autoppl_warmup = np.array([
    0.034373,
    0.312257,
    0.226581,
    1.61062,
    2.26579,
    5.40673,
    7.84947
])

autoppl_sampling = np.array([
    0.156954,
    0.297875,
    0.469138,
    1.04633,
    1.40626,
    2.41843,
    3.91615
])

autoppl_total = autoppl_warmup + autoppl_sampling

autoppl_esss = np.array([

    [25.4632,
    58.1867,
    25.7368,
    33.8489,
    79.4676],

    [1077.73,
    507.368,
    458.485,
    585.227,
    1295.52],

    [870.681,
    885.037,
    928.968,
    1283.45,
    1990.11],

    [2253.88,
    1883.97,
    1918.14,
    2840.06,
    3680.44],

    [3729.86,
     2846.7,
    3399.52,
    3781.89,
    6569.34],

    [11870.3,
    9123.73,
    10004.5,
    12639.1,
      19528],

    [15300.6,
      13206,
    13746.3,
    15566.8,
    25664.5]
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

import matplotlib.pyplot as plt

plt.style.use('dark_background')

x = [2**n for n in range(17)]

y_2_seq = []
y_2_cpu = []
y_2_seq_proj = []
y_2_cpu_proj = []
y_2_gpu = []

with open(f'tests/output.txt', 'r') as file:
    lines = file.readlines()

    for i in range(0, 9):
        y_2_seq.append(float(lines[i]) / 1000)

    for i in range(9, 14):
        y_2_cpu.append(float(lines[i]) / 1000)

    for i in range(14, 31):
        y_2_gpu.append(float(lines[i]) / 1000)

    file.close()

for i in range(9):
    if i == 0:
        y_2_seq_proj.append(y_2_seq[-1])
        y_2_cpu_proj.append(y_2_cpu[-1])
    else:
        y_2_seq_proj.append(y_2_seq_proj[-1] * 2)
        y_2_cpu_proj.append(y_2_cpu_proj[-1] * 2)

y_5_seq = []
y_5_cpu = []
y_5_seq_proj = []
y_5_cpu_proj = []
y_5_gpu = []

with open(f'tests/output1.txt', 'r') as file:
    lines = file.readlines()

    for i in range(0, 9):
        y_5_seq.append(float(lines[i]) / 1000)

    for i in range(9, 14):
        y_5_cpu.append(float(lines[i]) / 1000)

    for i in range(14, 31):
        y_5_gpu.append(float(lines[i]) / 1000)

    file.close()

for i in range(9):
    if i == 0:
        y_5_seq_proj.append(y_5_seq[-1])
        y_5_cpu_proj.append(y_5_cpu[-1])
    else:
        y_5_seq_proj.append(y_5_seq_proj[-1] * 2)
        y_5_cpu_proj.append(y_5_cpu_proj[-1] * 2)


y_15_seq = []
y_15_cpu = []
y_15_seq_proj = []
y_15_cpu_proj = []
y_15_gpu = []

with open(f'tests/output2.txt', 'r') as file:
    lines = file.readlines()

    for i in range(0, 6):
        y_15_seq.append(float(lines[i]) / 1000)

    for i in range(6, 8):
        y_15_cpu.append(float(lines[i]) / 1000)

    for i in range(8, 25):
        y_15_gpu.append(float(lines[i]) / 1000)

    file.close()

for i in range(12):
    if i == 0:
        y_15_seq_proj.append(y_15_seq[-1])
        y_15_cpu_proj.append(y_15_cpu[-1])
    else:
        y_15_seq_proj.append(y_15_seq_proj[-1] * 2)
        y_15_cpu_proj.append(y_15_cpu_proj[-1] * 2)




fig, axes = plt.subplots(3, 1, figsize = (7, 20), dpi = 200)


axes[0].plot(x[:9], y_2_seq, label='CPU (Sequential)', c='gray')
axes[0].plot(x[8:], y_2_seq_proj, linestyle='--', c='gray', alpha=0.3)
axes[0].plot(x[4:9], y_2_cpu, label='CPU (Parallel)', c='lightgray')
axes[0].plot(x[8:], y_2_cpu_proj, linestyle='--', c='lightgray', alpha=0.3)
axes[0].plot(x, y_2_gpu, label='GPU (Parallel)', c='orange')
axes[0].set_title('2 vs. 2')


axes[1].plot(x[:9], y_5_seq, label='CPU (Sequential)', c='gray')
axes[1].plot(x[8:], y_5_seq_proj, linestyle='--', c='gray', alpha=0.3)
axes[1].plot(x[4:9], y_5_cpu, label='CPU (Parallel)', c='lightgray')
axes[1].plot(x[8:], y_5_cpu_proj, linestyle='--', c='lightgray', alpha=0.3)
axes[1].plot(x, y_5_gpu, label='GPU (Parallel)', c='orange')
axes[1].set_title('5 vs. 5')

axes[2].plot(x[:6], y_15_seq, label='CPU (Sequential)', c='gray')
axes[2].plot(x[5:], y_15_seq_proj, linestyle='--', c='gray', alpha=0.3)
axes[2].plot(x[4:6], y_15_cpu, label='CPU (Parallel)', c='lightgray')
axes[2].plot(x[5:], y_15_cpu_proj, linestyle='--', c='lightgray', alpha=0.3)
axes[2].plot(x, y_15_gpu, label='GPU (Parallel)', c='orange')
axes[2].set_title('15 vs. 15')



for i in range(3):
    axes[i].set_xscale('log', base=2) 
    axes[i].set_xlabel('Number of Trajectories')
    axes[i].set_ylabel('Time (s)')
    # axes[i].set_ylim(0, 1000)
    axes[i].set_yscale('log', base=2) 
    axes[i].set_xticks(x)
    axes[i].grid(alpha=0.1)
    axes[i].legend()


plt.show()
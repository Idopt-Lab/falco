import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # or "Computer Modern"
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 16,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

# Load the CSV file
csv_file1 = 'power_req_trim_sim_results.csv'
csv_file2 = 'power_avail_trim_sim_results.csv'
csv_file3 = 'glide_perf_trim_sim_results.csv'
results_df1 = pd.read_csv(csv_file1)
results_df2 = pd.read_csv(csv_file2)
results_df3 = pd.read_csv(csv_file3)
print("Loaded results:")



# Plot Required Power vs VTAS
plt.figure()
plt.plot(results_df1['VTAS']* 1.944, results_df1['Required Power (kW)'], marker='s', linestyle='--', label='Required Power')
plt.plot(results_df2['VTAS']* 1.944, results_df2['Available Power (kW)'], marker='o', linestyle='-', label='Available Power')
plt.xlabel('KTAS (knots)')
plt.ylabel('Required Power (kW)')
plt.title('Required Power vs KTAS')
plt.grid(True)
plt.legend()
plt.show()

# Plot L/D Ratio vs VTAS
plt.figure()
KTAS = results_df1['VTAS'] * 1.944  # Convert m/s to knots
plt.plot(KTAS, results_df1['Lift']/results_df1['Drag'], marker='^', linestyle='--', label='Required Power L/D Ratio')
plt.plot(KTAS, results_df3['Lift']/results_df3['Drag'], marker='x', linestyle='-', label='Power Off Glide')
plt.xlabel('KTAS (knots)')
plt.ylabel('L/D Ratio')
plt.title('L/D Ratio vs KTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(results_df2['VTAS'], -results_df2['Climb Rate'], marker='^', linestyle='--', label='Rate of Climb')
plt.xlabel('VTAS (m/s)')
plt.ylabel('Rate of Climb (m/s)')
plt.title('Rate of Climb vs VTAS')
plt.grid(True)
plt.legend()
plt.show()


KTAS = results_df3['VTAS'] * 1.944  # Convert m/s to knots
FPM = KTAS * 101.269 # Convert knots to feet per minute
x1, y1 = 0, 0
x2, y2 = KTAS.max(), -925

m = (y2 - y1) / (x2 - x1)
b = y1 - m * x1

x_values = np.linspace(x1, x2, len(KTAS))
y_values = m * x_values + b

plt.figure(figsize=(6.5, 4.5), dpi=600)
mask = KTAS >= 88 # VSTALL from https://ntrs.nasa.gov/api/citations/20240010807/downloads/x57_performance_manuscript-v4.pdf
plt.plot(KTAS[mask], -FPM[mask] / ((results_df3['CL'] / results_df3['CD'])[mask]),
         marker='^', linestyle='--', label='Sink Rate')
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.vlines(128, ymin=-1000, ymax=-800, colors='black', linestyles='--', label=r'$V_{BG}$')
plt.xlabel('KTAS (knots)')
plt.ylim(-1000, 0)
plt.ylabel('Sink Rate (ft/min)')
# plt.title('Sink Rate vs KTAS')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Sink_Rate_vs_ktas.pdf", format='pdf', bbox_inches='tight')
plt.show()




plt.figure(figsize=(6.5, 4.5), dpi=600)
KTAS = results_df1['VTAS'] * 1.944  # m/s to knots
ld_curve1 = results_df1['CL'] / results_df1['CD']
ld_curve2 = (results_df1['CL']**(3/2)) / results_df1['CD']
Vbe_x = KTAS.iloc[ld_curve2.idxmax()]
Vbr_x = KTAS.iloc[ld_curve1.idxmax()]

plt.plot(KTAS, ld_curve1, marker='o', linestyle='-', color='blue', label=r"$\frac{C_{L}}{C_{D}}$")
plt.plot(KTAS, ld_curve2, marker='s', linestyle='-', color='red', label=r"$\frac{C_{L}^{3/2}}{C_{D}}$")
plt.vlines(Vbr_x, ymin=0, ymax=max(ld_curve1), colors='blue', linestyles='--', label=r"$V_{\mathrm{BR}}$")
plt.vlines(Vbe_x, ymin=0, ymax=max(ld_curve2), colors='red', linestyles='--', label=r"$V_{\mathrm{BE}}$")

plt.xlabel(r"KTAS (knots)")
plt.ylabel(r"$\frac{C_{L}}{C_{D}}$")
# plt.title(r"$\frac{C_{L}}{C_{D}}$ vs KTAS")

plt.grid(True, linestyle=':', linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("clcd_vs_ktas.pdf", format='pdf', bbox_inches='tight')
plt.show()







import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
plt.plot(results_df1['VTAS'], results_df1['Required Power (kW)'], marker='s', linestyle='--', label='Required Power')
plt.plot(results_df2['VTAS'], results_df2['Available Power (kW)'], marker='o', linestyle='-', label='Available Power')
plt.xlabel('VTAS (m/s)')
plt.ylabel('Required Power (kW)')
plt.title('Required Power vs VTAS')
plt.grid(True)
plt.legend()
plt.show()

# Plot L/D Ratio vs VTAS
plt.figure()
plt.plot(results_df1['VTAS'], results_df1['CL']/results_df1['CD'], marker='^', linestyle='--', label='Required Power L/D Ratio')
plt.plot(results_df3['VTAS'], results_df3['CL']/results_df3['CD'], marker='x', linestyle='-', label='Power off Glide')
plt.xlabel('VTAS (m/s)')
plt.ylabel('L/D Ratio')
plt.title('L/D Ratio vs VTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(results_df2['VTAS'], -results_df2['Rate of Descent'], marker='^', linestyle='--', label='Rateof Climb')
plt.xlabel('VTAS (m/s)')
plt.ylabel('Rate of Climb (m/s)')
plt.title('Rate of Climb vs VTAS')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.plot(results_df3['VTAS'], -results_df3['VTAS']/(results_df3['CL'] / results_df3['CD']), marker='^', linestyle='--', label='Sink Rate')
plt.plot(results_df3['VTAS'], results_df3['Minimum Sink Rate'], marker='^', linestyle='--', label='Minimum Sink Rate')
plt.xlabel('VTAS (m/s)')
plt.ylabel('Sink Rate (m/s)')
plt.title('Sink Rate vs VTAS')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
# Plot CL/CD Ratio vs VTAS
ld_curve1 = results_df1['CL'] / results_df1['CD']
ld_curve2 = (results_df1['CL']**(3/2)) / results_df1['CD']
Vbe_x = results_df1['VTAS'].iloc[ld_curve1.idxmax()]
Vbr_x = results_df1['VTAS'].iloc[ld_curve2.idxmax()]
plt.plot(results_df1['VTAS'], ld_curve1, marker='o', linestyle='-', label=r'$\frac{C_{L}}{C_{D}}$')
plt.plot(results_df1['VTAS'], ld_curve2, marker='s', linestyle='-', label=r'$\frac{C_{L}^\frac{3}{2}}{C_{D}}$')
plt.axvline(x=Vbe_x, color='red', linestyle='--', label='Best Endurance VTAS')
plt.axvline(x=Vbr_x, color='green', linestyle='--', label='Best Range VTAS')
plt.xlabel('VTAS (m/s)')
plt.ylabel('CL/CD Ratio')
plt.title('CL/CD Ratio vs VTAS')
plt.grid(True)
plt.legend()
plt.show()


csv_file4 = 'alpha_vs_CL_WITHOUT.csv'
results_df4 = pd.read_csv(csv_file4)
csv_file5 = 'alpha_vs_CL_WITH.csv'
results_df5 = pd.read_csv(csv_file5)

plt.figure()
plt.plot(results_df4['Alpha'] * 180/np.pi, results_df4['CL'], marker='o', linestyle='-', label='Without Blowers')
plt.plot(results_df5['Alpha'] * 180/np.pi, results_df5['CL'], marker='s', linestyle='--', label='With Blowers')
plt.xlabel('Angle of Attack (degrees)')
plt.ylabel('Lift Coefficient (CL)')
plt.title('Lift Coefficient vs Angle of Attack')
plt.grid(True)
plt.legend()
plt.show()




# X-57 Optimizations used in the Aviation Forum 2025 paper
----

This file explains how to produce the plots used in the Aviation Forum 2025 paper. Below outlines the simple process for any of the optimizations:

* Open the optimizations folder
* Select any of the optimization files you want to look at and then run it
	* The results are stored in the X-57/optimizations/results folder as .csv files
* When you are finished an optimization, open the plotting.py file to view the plots shown in the Aviation Forum 2025 paper

----

Table 7: Different Steady-Level Cruise Flight Conditions Trim Settings results are generated using the x57_trim_FM.py file

Table 8: X-57 Takeoff V-Speed optimizations are generated from the following files:
* v_stall_TO_trim_opt.py
* v_X_TO_trim_opt.py
* v_Y_TO_trim_opt.py
* v_R_TO_trim_opt.py
* v_MU_TO_trim_opt.py

Figure 15: X-57 Best Range and Endurance Velocities at 8,000 ft results are generated using the trim_power_req_min_opt.py file

Figure 16: X-57 Sink Rate plotted against KTAS at 8,000 ft results are generated using trim_power_avail_max_opt.py and trim_glide_performance.py files

The maximum horizontal velocity value in Table 9 is found using the v_H_cruise_trim.py file


Enjoy!

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 22:04:10 2025

@author: justin france
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import control.matlab as control
import pandas as pd


DAMPENING = 0.1     # required dampening  more than
E_SS = 0.0001           # steady state error less than[mm].
M_P = 30            # overshoot less than [%]
T_R = 0.5            # Rise time less than seconds [s]




    
def extract_data ( filepath ): # Put the filename / filepath of your datafile here
    # INCLUDING the . mat suffix
   data = np.loadtxt(filepath, delimiter=",", skiprows=35, usecols=(1,2,3,4,5,6,7,8))
   Time = data[:,0]
   System_responce = data[:,1]
   Voltage = data[:, 2]
   Step_input = data[:, 3]
   RMV_sencor = data[:,4]
   k_d_gain_list = data[:,5]
   k_i_gain_list = data[:,6]
   k_p_gain_list = data[:,7]
   
   # print("kd_gain", data[:,5])
   # print("ki_gain",data[:,6])
   # print("kp_gain",data[:,7])
   # print("time=", Time, "\nSystem_responce=",System_responce, '\nVoltage=',Voltage,"\nRMV_sencor=", RMV_sencor, '\nstep_input=', Step_input)
   return Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list  

    
def find_damping_ratio_from_specs(M_P):
    damping = np.linspace(0.00, 0.99, 1000)  # Avoid 1 to stay within sqrt domain
    matched_damping = []

    for i in damping:
        try:
            val = 100 * np.exp(-(np.pi * i) / np.sqrt(1 - i**2))
            if np.isclose(M_P, val, atol=0.1):  # Adjust tolerance as needed
                matched_damping.append(i)
        except:
            continue

    return matched_damping[0]

def get_poles_from_file(BETA, C, filepath):    
    """finding plotting pole locations for the experiments"""
    Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(filepath)
    
    if len(k_p_gain_list) == 0:
       print(f"[ERROR] No kp_gain data in file: {filepath}")
       return None, "UNKNOWN"
   
    kp_gain = k_p_gain_list[0]
    kd_gain = k_d_gain_list[0]
    ki_gain = k_i_gain_list[0]
    
    control_type = ""
    poles = None
    
    if(kp_gain != 0 and kd_gain == 0 and ki_gain == 0):
        control_type = "P"
        sys = control.tf([0, 0, 0.1*BETA * kp_gain], [1, C, kp_gain * BETA])
        poles = control.pole(sys)
        print("\nP control pole locations kp=", kp_gain, "\nP control pole locations", poles,"\n")
        
    elif(kp_gain != 0 and ki_gain == 0 and kd_gain != 0):
         control_type = "PD"
         sys = control.tf([0, 0.1* BETA * kd_gain, 0.1 * BETA * kp_gain], [1, C + BETA * kd_gain, kp_gain * BETA])
         poles = control.pole(sys)
         print("\nkp=", kp_gain, "kd", kd_gain,"\nPd control pole locations", poles,"\n")
             
    elif(kp_gain != 0 and ki_gain != 0 and kd_gain != 0):
         control_type = "PID"
         sys = control.tf([0.1 * BETA * kd_gain, 0.1 * BETA * kp_gain ,  0.1 * BETA * ki_gain ], [1, C + BETA * kd_gain, kp_gain * BETA, ki_gain * BETA])
         [y, t] = control.step(sys)
         poles = control.pole(sys)
         print("\nkp=", kp_gain, "ki=",ki_gain, "kd=", kd_gain, "\nPID control pole locations", poles,"\n") 
         
    return poles, control_type


def design_region_blode_plot(DAMPENING, E_SS, M_P, T_R, min_damping_ratio, BETA, C, filepath):
    """ find the design region for a blode plot from provided specs"""
    
    """ caculation from design specs"""
    w_n = 1.8 / T_R
    theta = np.arcsin(min_damping_ratio)
    Re = -w_n * min_damping_ratio
    Im = np.sqrt(w_n**2  - (Re)**2)
    Poles = complex (Re, Im)
    
    print("w_n <=", w_n, "[Rad /s]")
    print(f'min_damping", {min_damping_ratio:.3f}')
    print("theta=", theta, "[rad]")
    print(f"minimum poleslocation =", Poles, "\n\n")

    
    """ plotting the pole locations from experiments"""
    fig, ax = plt.subplots(figsize=(3, 8))
    
    for file in filepath:
        poles, control_type = get_poles_from_file(BETA, C, file)
        if poles is not None:
            filtered_poles = poles[np.real(poles) >= -10]
            if control_type =="P":
                ax.plot(np.real(filtered_poles), np.imag(filtered_poles), 'rx', label='Kp simulated Poles')
            elif control_type =="PD":
                ax.plot(np.real(filtered_poles), np.imag(filtered_poles), 'bx', label='Kd simulated Poles')
            elif control_type =="PID":
                ax.plot(np.real(filtered_poles), np.imag(filtered_poles), 'mx', label='Ki simulated Poles')
    
    theta_vals = np.linspace(0, np.pi, 500)  # Semicircle from 0 to pi
    x_vals = -w_n * np.cos(theta_vals - np.pi/2)  # x = -w_n * cos(theta)
    y_vals = w_n * np.sin(theta_vals - np.pi/2)  # y = w_n * sin(theta)
    ax.plot(x_vals, y_vals, 'g--', label='Rise time')  # Plot the semicircle
    
    # Damping line
    x = np.linspace(-2 * w_n, 0, 500)
    ax.plot(x, np.tan(theta - np.pi / 2) * x, 'b--', linewidth=0.5, label="Over shoot")
    ax.plot(x, -np.tan(theta - np.pi / 2) * x, 'b--', linewidth=0.5)
     
  
    # Axes styling
    ax.spines['right'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    
    ax.xaxis.set_ticks(np.linspace(-10, 10, 11))
    ax.yaxis.set_ticks(np.linspace(-18, 18, 19))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('right')
            
    # Custom axis labels positioned at the ends of axes
    ax.set_xlabel("")  # Remove default x-label
    ax.set_ylabel("")  # Remove default y-label
    # Get axis limits to place the labels at the ends
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    # Add custom labels at the end of axes
    ax.text(x_lim[1], 0.05, "Re", ha='left', va='bottom', fontsize=12)
    ax.text(-0.05, y_lim[1] * 0.90, "Im", ha='right', va='top', fontsize=12)
    
    ax.grid(True, which='both')
    ax.axis('equal')
   
    
    ax.legend(loc='upper left', bbox_to_anchor=(0.70, 0.95), fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(left=-1, right=1)
    plt.show()
    
    
def proportional_control_sys_output_TF_simulated(BETA, C, min_damping_ratio, E_SS):
    """caculate possiav]ble values for kp"""

    kp_val = np.linspace(1, 131, 130)
    kp_max = "n/a"
    for kp in kp_val:
        sys_dampening = C / (2 *np.sqrt(BETA * kp))
        if min_damping_ratio <= sys_dampening:
            kp_max = kp
        # if min_damping_ratio >= sys_dampening:
        #     #print(min_damping_ratio,"<=", sys_dampening, "valid values of kp =", kp)
        #     kp_max = kp
        #     print("Invalid values for kp=", kp_max)
    print("\n max value of kp =", kp_max)
    
    axes = plt.axes()
    kp_gains = np.linspace(25, 125, 21)
    
    for kp_gain in kp_gains:
        sys = control.tf([0, 0, 0.1 * BETA * kp], [1, C, kp_gain * BETA])
        #print(sys)
        [y, t] = control.step(sys)
        axes.plot(t,y, label=f"kp= {kp_gain:.1f}")
        print("\n P control pole locations kp=", kp_gain, "\n", control.pole(sys),"\n")
        
        
    axes.grid(which="major", color="black", linewidth=0.9)
    axes.grid(which="minor", color="black", linewidth=0.3)
    axes.minorticks_on()
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Amplitude x")
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=5)
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Response [m]")
    plt.show()
    
   
def derivitive_control_sys_output_TF_simulated(BETA, C, min_damping_ratio):
    """ caculate possiable values for kd"""
    kp= 100
                             # choose the approprate kp value from proportional control
    kd_val = np.linspace(0, 20, 5)
    kd_max = "n/a"
    for kd in kd_val:
        sys_dampening = (C + BETA * kd) / (2 *np.sqrt(BETA * kp))
        if min_damping_ratio > sys_dampening:
            #print(min_damping_ratio,"<=", sys_dampening, "valid values of kd =", kd)
            kd_max = kd
    print("min value for kd =", kd_max)
    
    axes = plt.axes()
    for kd_gain in kd_val:
        #print(kd)
        sys = control.tf([0, 0.1 * BETA * kd_gain, 0.1 * BETA * kp], [1, C + BETA * kd_gain, kp * BETA])
        [y, t] = control.step(sys)
        print("\n PD control pole locations kp=", kp, "kd=", kd_gain, "\n", control.pole(sys),"\n")
        axes.plot(t,y, label=f"kd= {kd_gain:.1f}")
        
    axes.grid(True)
    axes.legend(loc="lower right" )
    axes.grid(which="major", color="black", linewidth=0.9)
    axes.grid(which="minor", color="black", linewidth=0.3)
    axes.minorticks_on()
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Response [m]")
    plt.show()
    
    
def intergral_control_sys_output_TF_simulated(BETA, C, min_damping_ratio):
    """ caculate possiable values for kd"""
    kd = 15
    kp = 100 
    axes = plt.axes()
    ki_val = np.linspace(0, 140, 6)
    #ki_val = np.linspace(180, 250, 3)
    for ki in ki_val:
         #print(ki)
         sys = control.tf([0.1 * BETA * kd, 0.1 * BETA * kp ,  0.1 * BETA * ki ], [1, C + BETA * kd, kp * BETA, ki * BETA])
         [y, t] = control.step(sys)
         print("\n PID control pole locations kp=", kp, "ki=",ki, "kd=", kd, "\n", control.pole(sys),"\n")
         axes.plot(t,y,  label=f"ki= {ki:.1f}")
    axes.grid(True)
    axes.legend(loc="lower right" )
    axes.grid(which="major", color="black", linewidth=0.9)
    axes.grid(which="minor", color="black", linewidth=0.3)
    axes.minorticks_on()
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Response [m]")
    plt.show()
    
    
def plot_real_sys_responce(Time, system_responce, Step_input):
    """ plot the realk system responce"""
    axes =plt.axes()
    axes.plot(Time, system_responce, label="System responce")
    axes.plot(Time, Step_input, label="Step input")
    axes.legend(loc="lower right" )
    axes.set_xlabel("Time [s]")
    axes.set_ylabel("Responce [m]")
    axes.grid(True)
    plt.show()
    
    
def  plotting_real_and_simulated_responce(Time,
                                      System_responce, 
                                      Voltage, 
                                      Step_input, 
                                      RMV_sencor, 
                                      k_d_gain_list, 
                                      k_i_gain_list, 
                                      k_p_gain_list, 
                                      BETA, 
                                      C):
    
    """ plots the real world data agenst the starting values for the simulated results for P, PD, PID control"""
    
    """determining control type"""
    kp_gain = k_p_gain_list[0]
    kd_gain = k_d_gain_list[0]
    ki_gain = k_i_gain_list[0]
    kp_poles = np.array([])
    kd_poles = np.array([])
    ki_poles = np.array([])
    
    control_type = ""
    control_responce_data = np.array([])
    time_data = np.array([])
    
    if(np.all(k_p_gain_list) != 0 and np.all(k_i_gain_list) == 0 and np.all(k_d_gain_list) == 0):
        control_type = "P"
        
        sys = control.tf([0, 0, 0.1*BETA * kp_gain], [1, C, kp_gain * BETA])
        [y, t] = control.step(sys)
        time_data = t
        control_responce_data = y
        kp_poles = control.pole(sys)
        w_n = np.sqrt(np.real(kp_poles)[1]** 2 + np.imag(kp_poles)[1] **2)
        damping = np.abs(np.real(kp_poles)[0] / w_n)
        ess = max(Step_input) - y[-1]
        
        print("\nP control pole locations simulated: kp=", kp_gain, "\nP control pole locations", kp_poles)
        print("maximium value of simulated responce=", max(y))
        print("step_responce=", max(Step_input),"\n")
        print(f"Simulated: w_n= {w_n:.3f} [rad/s]")
        print(f"Simulated: damping= {damping:.3f}\n")
        print(f"Simulated: MP= {((max(y) - y[-1]) * 1000):.2f}%")
        print(f"Simulated: ts= {(4.6 / np.abs(np.real(kp_poles)[0])):.2f} [seconds]")
        
        print(f"Simulated tr= {(1.8/ w_n):.2f} [seconds]")
        print(f"Simulated ess= {ess:.5f} [m]\n")
        
        
    elif(np.all(k_p_gain_list) != 0 and np.all(k_i_gain_list) == 0 and np.all(k_d_gain_list) != 0):
         control_type = "PD"
         sys = control.tf([0, 0.1* BETA * kd_gain, 0.1 * BETA * kp_gain], [1, C + BETA * kd_gain, kp_gain * BETA])
         [y, t] = control.step(sys)
         time_data = t
         control_responce_data = y
         kd_poles = control.pole(sys)
         w_n =np.sqrt(np.real(kd_poles)[0]** 2 + np.imag(kd_poles)[0] **2)
         damping = np.abs(np.real(kd_poles)[0] / w_n)
         ess = max(Step_input) - y[-1]
         
         tr_simulated = "Spec not meet"
         tr_point = max(Step_input) * 0.9
         if np.any(y >= tr_point):
             tr_simulated_index = np.where(y >= tr_point)[0][0]
             tr_sim = t[tr_simulated_index]
             tr_simulated = f"{t[tr_simulated_index]:.2f}"
         
         print("\nkp=", kp_gain, "kd", kd_gain,"\nPd control pole locations", kd_poles,"\n")
         print("maximium value of simulated responce=", max(y))
         print("step_responce=", max(Step_input),"\n")
         print(f"Simulated: w_n= {w_n:.3f} [rad/s]")
         print(f"Simulated: damping= {damping:.3f}\n")
         print(f"Simulated: MP= {((max(y) - y[-1]) * 1000):.2f}%")
         print(f"Simulated: ts= {(4.6 / np.abs(np.real(kd_poles)[1])):.2f} [seconds]")
         print(f"Simulated tr= {(tr_simulated)} [seconds]")
         print(f"Simulated ess= {ess:.5f} [m]\n")

    
    elif(np.all(k_p_gain_list) != 0 and np.all(k_i_gain_list) != 0 and np.all(k_d_gain_list) != 0):
         control_type = "PID"
         sys = control.tf([0.1 * BETA * kd_gain, 0.1 * BETA * kp_gain ,  0.1 * BETA * ki_gain ], [1, C + BETA * kd_gain, kp_gain * BETA, ki_gain * BETA])
         [y, t] = control.step(sys)
         time_data = t
         control_responce_data = y
         ki_poles = control.pole(sys)
         w_n =np.sqrt(np.real(ki_poles)[0]** 2 + np.imag(ki_poles)[1] **2)
         damping = np.abs(np.real(ki_poles)[0] / w_n)
         ess = max(Step_input) - y[-1]
         
         tr_simulated = "Spec not meet"
         tr_point = max(Step_input) * 0.9
         if np.any(y >= tr_point):
             tr_simulated_index = np.where(y >= tr_point)[0][0]
             tr_sim = t[tr_simulated_index]
             tr_simulated = f"{t[tr_simulated_index]:.2f}"
         
         print("\nkp=", kp_gain, "ki=",ki_gain, "kd=", kd_gain, "\nPID control pole locations", ki_poles,"\n")
         print("maximium value of simulated responce=", max(y))
         print("step_responce=", max(Step_input),"\n")
         print(f"Simulated: w_n= {w_n:.3f} [rad/s]")
         print(f"Simulated: damping= {damping:.3f}\n")
         print(f"Simulated: MP= {((max(y) - y[-1]) * 1000):.2f}%")
         print(f"Simulated: ts= {(4.6 / np.abs(np.real(ki_poles)[1])):.2f} [seconds]")
         print(f"Simulated tr= {tr_simulated} [seconds]")
         print(f"Simulated ess= {ess:.5f} [m]\n")
         
         
    """processing lab data"""     
    ti = np.where(Step_input != 0.0)[0][0] - 1
    end_time_index_reduced = np.where(Step_input[ti:] == 0.0)[0][1]
    tf = end_time_index_reduced + len(Step_input[:ti])
    time_range = Time[ti:tf] - Time[ti]
    responce_range = System_responce[ti:tf] 
    step_range = Step_input[ti:tf] - Step_input[ti]
    
    MP_real = (max(responce_range) -  responce_range[-1]) *1000
    ess_real = np.abs(max(Step_input) - responce_range[-1])
    
    tr_real = "Spec not meet"
    tr_point = max(Step_input) * 0.9
    if np.any(responce_range >= tr_point):
        tr_real_index = np.where(responce_range >= tr_point)[0][0]
        tr = time_range[tr_real_index]
        tr_real = f"{(time_range[tr_real_index]):.2f}"
        
    
    print(f"Real: MP= {MP_real:.2f}%")
    print(f"Real: tr= {tr_real} [seconds]")
    print(f"Real: ess= {ess_real:.5f} [m]\n")
   
    
    
    print(control_type, "control")
    print("start index=", ti, "\nend index=", end_time_index_reduced, "\nrange of values=", tf )
    
    axes = plt.axes()
    if np.any(time_data > 8):
        plt.xlim(0, 8)
        
    axes.plot(time_range, responce_range, label="Real cart response")
    axes.plot(time_data, control_responce_data, label="Simulated cart response")
    axes.plot(time_range, step_range, label="Step response")
    axes.plot
    axes.grid(which="major", color="black", linewidth=0.45)
    axes.grid(which="minor", color="black", linewidth=0.15)
    axes.minorticks_on()
    axes.set_xlabel("Time [s]")
    axes.set_ylabel(f"{control_type} control response [m]")
    axes.legend(loc="lower right")
    plt.show()
    
 
     
   
   
def main():
    """main fuction"""
    m_c = 1.5           # mass of cart [kg]
    k_m = 0.017         # back emf constant [V rad^-1 s^-1]
    k_g = 3.7           # gear ratio
    R = 1.5             # Risistance motor arnature [ohm]
    r_p = 0.018        # Radius of pinion (m)
    D = 7               # dampening of the system
    
    BETA = (k_m * k_g) / (m_c * R * r_p)                    # Back EMF from motor
    C = (D / m_c + (k_m**2 * k_g**2)/(m_c * R * r_p**2))    # Dampening on system
    
    
    filepath1 = ["mco203_p1.csv",
                "mco203_pd1.csv",
                "mco203_pid1.csv"]
    
    filepath3 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set3.csv"]
    filepath4 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set4.csv"]
    filepath5 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set5.csv"]
    filepath6 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set6.csv"]
    """ use file path 7 for report"""
    filepath7 = ["mco203_p1.csv", 
                 "mco203_pd1.csv",
                 "mco203_set7.csv"]
    filepath8 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set8.csv"]
    filepath9 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set9.csv"]
    filepath10 = ["mco203_set1.csv", 
                 "mco203_set2.csv",
                 "mco203_set10.csv"]
    
                
    p_cont_file_1A = "mco203_p1.csv"
    pd_cont_file_2A = "mco203_pd1.csv"
    pdi_cont_file_3A = "mco203_pid1.csv"
    
    
    p_cont_file_1 = "mco203_set1.csv"
    pd_cont_file_2 = "mco203_set2.csv"
    pid_cont_file_3 = "mco203_set3.csv"
    pid_cont_file_4 = "mco203_set4.csv"
    pid_cont_file_5 = "mco203_set5.csv"
    pid_cont_file_6 = "mco203_set6.csv"
    pid_cont_file_7 = "mco203_set7.csv"
    pid_cont_file_8 = "mco203_set8.csv"
    pid_cont_file_9 = "mco203_set9.csv"
    pid_cont_file_10 = "mco203_set10.csv"
    
    
    """ data from lab 1 19/2/25"""
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(p_cont_file_1A)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pd_cont_file_2A)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pdi_cont_file_3A)
    
    """ data fron lab 2 26/3/25""" 
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(p_cont_file_1)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pd_cont_file_2)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_3)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_4)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_5)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_6)
    """ use file 7 for report"""
    Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_7)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_9)
    #Time, System_responce, Voltage, Step_input, RMV_sencor, k_d_gain_list , k_i_gain_list, k_p_gain_list = extract_data(pid_cont_file_10)
    
    # plot_real_sys_responce(Time, System_responce, Step_input)
    #print(Time)
    #print(System_responce)
    
    
    """ Plotting_real_data_and_simulated"""
    plotting_real_and_simulated_responce(Time,
                                          System_responce, 
                                          Voltage, 
                                          Step_input, 
                                          RMV_sencor, 
                                          k_d_gain_list, 
                                          k_i_gain_list, 
                                          k_p_gain_list, 
                                          BETA, 
                                          C)
    
    
    """design region""" 
    min_damping_ratio = find_damping_ratio_from_specs(M_P)
    
    get_poles_from_file(BETA, C, filepath7) 
    design_region_blode_plot(DAMPENING, E_SS, M_P, T_R, min_damping_ratio, BETA, C, filepath7)
    
    """ simulated plots """
    #proportional_control_sys_output_TF_simulated(BETA, C, min_damping_ratio, E_SS)
    
    #derivitive_control_sys_output_TF_simulated(BETA, C, min_damping_ratio)
    
    #intergral_control_sys_output_TF_simulated(BETA, C, min_damping_ratio)
    
   
    
main()
    


    

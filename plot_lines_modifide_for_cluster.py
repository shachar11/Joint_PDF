#! /usr/bin/env python

"""
Script for plotting 1D data from .athdf, .hst, or .tab files.

Run "plot_lines.py -h" to see full description of inputs.

Multiple lines can be plotted by having any of the first three arguments be
comma-separated lists. If one list runs out before another, its last entry will
be repeated as necessary.

Use "show" for the 4th argument to show interactive plot instead of saving to
file.
"""

# Python modules
import argparse
import re

# Athena++ modules
import athena_read

import numpy as np

# Main function
def main(**kwargs):

    # Extract inputs
    data_files = kwargs['data_files'].split(',')
    x_names = kwargs['x_names'].split(',')
    y_names = kwargs['y_names'].split(',')
    output_file = kwargs['output_file']
    styles = kwargs['styles'].split(',')
    colors = kwargs['colors']
    labels = kwargs['labels']
    x_log = kwargs['x_log']
    y_log = kwargs['y_log']
    x_min = kwargs['x_min']
    x_max = kwargs['x_max']
    y_min = kwargs['y_min']
    y_max = kwargs['y_max']
    x_label = kwargs['x_label']
    y_label = kwargs['y_label']
    profile_name=kwargs['profile_name']
    

    # Verify inputs
    num_lines = max(len(data_files), len(x_names), len(y_names))
    if data_files[0] == '':
        raise RuntimeError('First entry in data_files must be nonempty')
    if x_names[0] == '':
        raise RuntimeError('First entry in x_names must be nonempty')
    if y_names[0] == '':
        raise RuntimeError('First entry in y_names must be nonempty')
    if len(data_files) < num_lines:
        data_files += data_files[-1:] * (num_lines - len(data_files))
    if len(x_names) < num_lines:
        x_names += x_names[-1:] * (num_lines - len(x_names))
    if len(y_names) < num_lines:
        y_names += y_names[-1:] * (num_lines - len(y_names))
    for n in range(num_lines):
        if data_files[n] == '':
            data_files[n] = data_files[n-1]
        if x_names[n] == '':
            x_names[n] = x_names[n-1]
        if y_names[n] == '':
            y_names[n] = y_names[n-1]
    for data_file in data_files:
        valid_file = (data_file[-6:] == '.athdf' or data_file[-4:] == '.hst'
                      or data_file[-4:] == '.tab')
        if not valid_file:
            raise RuntimeError('Files must have .athdf, .hst, or .tab extension')
    if len(styles) < num_lines:
        styles += styles[-1:] * (num_lines - len(styles))
    for n in range(num_lines):
        styles[n] = styles[n].lstrip()
        if styles[n] == '':
            styles[n] = '-'
    if colors is None:
        colors = [None] * num_lines
    else:
        colors = colors.split(',')
        if len(colors) < num_lines:
            colors += colors[-1:] * (num_lines - len(colors))
        for n in range(num_lines):
            if colors[n] == '':
                colors[n] = None
    if num_lines == 1 and colors[0] is None:
        colors[0] = 'k'
    if labels is None:
        labels = [None] * num_lines
    else:
        labels = labels.split(',')
        if len(labels) < num_lines:
            labels += [None] * (num_lines - len(labels))
        for n in range(num_lines):
            if labels[n] == '':
                labels[n] = None
    labels_used = False
    for n in range(num_lines):
        if labels[n] is not None:
            labels_used = True
            break

    # Load Python plotting modules
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # Read data
    x_vals = []
    y_vals = []
    
    
    #intial paramters in cgs
    P0=1.01325e+06 
    T0=298
    Ru_cgs=8.3144*10**7
    rho0 = P0 / (Ru_cgs * T0)

    #read data from hdf5 files
    for n in range(num_lines):
        if data_files[n][-6:] == '.athdf':
            data = athena_read.athdf(data_files[n])
        elif data_files[n][-4:] == '.hst':
            data = athena_read.hst(data_files[n])
        else:
            data = athena_read.tab(data_files[n])

        # Assuming data is your existing dictionary
        data_new = {
            "Pressure": np.array(data["Pressure"]),
            "Temperature": np.array(data["Temperature"]),
            "Fuel-MassFraction": np.array(data["Fuel-MassFraction"]),
            "Density": np.array(data["Density"])
        }
        if any(value is None for value in data_new.values()):
            raise ValueError("Some keys in the data are missing values.")
        
        skip_points = 100
        # Create a mask where Fuel-MassFraction is not zero
        mask = data_new["Fuel-MassFraction"] > 0.01  # This is a 3D mask (same shape as Fuel-MassFraction)

        # Apply the mask to filter out all lists with corresponding zero values in Fuel-MassFraction
        data_filtered_MassFracion_not_0 = {
            key: val[mask]  # Filter using the mask on all keys
            for key, val in data_new.items()
        }

        x_vals.append(data_filtered_MassFracion_not_0[x_names[n]]) #[::skip_points])
        y_vals.append(data_filtered_MassFracion_not_0[y_names[n]]) #[::skip_points])


        # x_vals.append(data[x_names[n]].flatten()) #[::skip_points])
        # y_vals.append(data[y_names[n]].flatten()) #[::skip_points])
    

    ### reading ZND file
    # Define arrays to store flow field data
    x1d, P, rho, e, T, U, c, X, Ma, Ma_eq = [], [], [], [], [], [], [], [], [], []

    # Counter for number of points
    NPTS = 0

    # Open and read the file
    with open(f"/kozaky_ssd/charash/athenaplusplus/src/science/detonation/profiles/adiabatic/detonation.{profile_name}.dat", "r") as file:
        for line in file:
            # Skip comment lines (starting with #)
            if line.startswith("#"):
                continue
        
            # Skip empty lines
            if line.strip() == "":
                continue
        
            # Split the line into tokens and convert to float
            tokens = [float(value) for value in line.split()]
        
            # Store flow field data in cgs units
            x1d.append(tokens[0] * 100.0)
            P.append(tokens[1] * 10)
            rho.append(tokens[2] * 0.001)
            e.append(tokens[3] * 1.0)
            T.append(tokens[4] * 1.0)
            U.append(tokens[5] * 100)
            c.append(tokens[6] * 100)
            X.append(1 - tokens[7])
            Ma.append(U[NPTS] / c[NPTS])
            Ma_eq.append(U[NPTS] / c[NPTS])

            # Increment the point counter
            NPTS += 1

############################################################################
    
    def extract_float_after_G(input_string):
        match = re.search(r'G_([-+]?\d*\.?\d+)', input_string)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("The input string does not contain 'G_' followed by a float.")
    
    def extract_float_after_Q(input_string):
        match = re.search(r'Q_([-+]?\d*\.?\d+)', input_string)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("The input string does not contain 'Q_' followed by a float.")
        
    q_norm=extract_float_after_Q (profile_name)  
    gamma=extract_float_after_G(profile_name)
    MW=27 #cgs
    R_cgs=8.3144*10**7/MW
    rho0 = P0 / (R_cgs * T0)
    Q = q_norm * R_cgs* T0
    qggg = Q / P0 * rho0 * (gamma ** 2 - 1.) / (2. * gamma)
    M=(np.sqrt(1. + qggg) + np.sqrt(qggg))

   
    Mach08=M*0.8
    Mach12=M*1.2
    T08_T0 = (((2*gamma*(Mach08**2)+1-gamma)*((Mach08**2)*(gamma-1)+2))/(((gamma+1)**2)*(Mach08**2)))
    T12_T0 = (((2*gamma*(Mach12**2)+1-gamma)*((Mach12**2)*(gamma-1)+2))/(((gamma+1)**2)*(Mach12**2)))
    P08_P0 = (1 + (2 * gamma / (gamma + 1)) * (Mach08**2 - 1))
    P12_P0 = (1 + (2 * gamma / (gamma + 1)) * (Mach12**2 - 1))
    rho08_rho0=P08_P0/T08_T0
    rho12_rho0=P12_P0/T12_T0


############################################################################
    # normilize values
    P=[p/P0 for p in P]
    T=[t/T0 for t in T]
    Dens=[rho0/q for q in rho]

    all_x_values=[]
    all_y_values=[]

    for n in range(num_lines):
        all_x_values.extend(x_vals[n])
        all_y_values.extend(y_vals[n])

    if x_names[0] == 'Pressure':
        all_x_values = [x / P0 for x in all_x_values]
        X_axis=P
        x1=P08_P0
        x2=P12_P0
    elif x_names[0] =='Temperature':
        all_x_values = [x / T0 for x in all_x_values]
        X_axis=T
        x1=T08_T0
        x2=T12_T0        
    elif x_names[0] =='Density':
        all_x_values = [rho0/x for x in all_x_values]
        X_axis=Dens
        x1=1/rho08_rho0
        x2=1/rho12_rho0   
    if  y_names[0] == 'Pressure':
        all_y_values = [y / P0 for y in all_y_values]
        Y_axis=P
        y1=P08_P0
        y2=P12_P0
    elif y_names[0] =='Temperature':
        all_y_values = [y / T0 for y in all_y_values]
        Y_axis=T
        y1=T08_T0
        y2=T12_T0
    elif y_names[0] =='Density':
        all_y_values = [rho0/y for y in all_y_values]
        Y_axis=Dens
        y1=1/rho08_rho0
        y2=1/rho12_rho0

    #do the satistic part
    num_bins = 100
    joint_pdf, xedges, yedges = np.histogram2d(all_x_values,all_y_values, bins=num_bins, density=True)
    joint_pdf[joint_pdf == 0] = np.nan
    joint_pdf=np.log(joint_pdf)

    # Plot the joint PDF
    plt.figure()
    plt.imshow(joint_pdf.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='inferno') 
    plt.colorbar(label='Probability Density') 

    plt.scatter(X_axis, Y_axis, color='red', s=2) 

    if gamma is not None and q_norm is not None:
        plt.plot([x1, x2], [y1, y2], linestyle='--', linewidth=0.5, color='red')

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    #if labels_used:
    #    plt.legend(loc='best')

    if output_file == 'show':
        plt.show()
    else:
        plt.savefig(output_file+'_joint_pdf.png', bbox_inches='tight')
    plt.close()

    # Plot data
    # plt.figure()
    # for n in range(num_lines):
    #     plt.plot(x_vals[n], y_vals[n], styles[n], color=colors[n], label=labels[n])
    # if x_log:
    #     plt.xscale('log')
    # if y_log:
    #     plt.yscale('log')
    # plt.xlim((x_min, x_max))
    # plt.ylim((y_min, y_max))
    # if x_label is not None:
    #     plt.xlabel(x_label)
    # if y_label is not None:
    #     plt.ylabel(y_label)
    # if labels_used:
    #     plt.legend(loc='best')
    # if output_file == 'show':
    #     plt.show()
    # else:
    #     plt.savefig(output_file+'.png', bbox_inches='tight')
    # plt.close()

# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'data_files',
      help=('comma-separated list of input files; empty strings repeat previous entries; '
            'list is extended if x_names or y_names is longer')
    )
    parser.add_argument(
        'x_names',
        help=('comma-separated list of abscissas; empty strings repeat previous entries; '
              'list is extended if data_files or y_names is longer')
    )
    parser.add_argument(
        'y_names',
        help=('comma-separated list of ordinates; empty strings repeat previous entries; '
              'list is extended if data_files or x_names is longer')
    )
    parser.add_argument(
        'output_file',
        help=('name of output to be (over)written; use "show" to show interactive plot '
              'instead')
    )
    parser.add_argument(
      '-s', '--styles',
      default='-',
      help=('comma-separated list of line or marker styles, such as "-" or "o"; use the '
            '" -s=..." form of the argument if the first entry begins with a dash; empty '
            'strings are interpreted as solid lines; last entry is repeated as necessary')
    )
    parser.add_argument(
      '-c', '--colors',
      help=('comma-separated list of color codes, such as "k", "blue", or "#123abc"; '
            'empty strings result in black (single line) or default color cycling '
            '(multiple lines); last entry is repeated as necessary')
    )
    parser.add_argument(
      '-l', '--labels',
      help=('comma-separated list of labels for legend; empty strings are not added to '
            'legend; strings can include mathematical notation inside $...$ (e.g. "-l '
            '\'$\\rho$\'")')
    )
    parser.add_argument(
        '--x_log',
        action='store_true',
        help='flag indicating x-axis should be log scaled')
    parser.add_argument(
        '--y_log',
        action='store_true',
        help='flag indicating y-axis should be log scaled')
    parser.add_argument('--x_min',
                        type=float,
                        help='minimum for x-axis')
    parser.add_argument('--x_max',
                        type=float,
                        help='maximum for x-axis')
    parser.add_argument('--y_min',
                        type=float,
                        help='minimum for y-axis')
    parser.add_argument('--y_max',
                        type=float,
                        help='maximum for y-axis')
    parser.add_argument('--x_label',
                        help='label to use for x-axis')
    parser.add_argument('--y_label',
                        help='label to use for y-axis')
    parser.add_argument('--profile_name',
                        help='for plotting ZND')
    
    args = parser.parse_args()
    main(**vars(args))

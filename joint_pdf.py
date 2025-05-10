#! /usr/bin/env python

"""
Script for plotting 1D data from .athdf, .hst, or .tab files.

Run "plot_lines.py -h" to see full description of inputs.
"""

# Python modules
import argparse

# Athena++ modules
import athena_read

import numpy as np

# Main function
def main(**kwargs):

    # Extract inputs
    data_file = kwargs['data_files']  # Single data file
    x_name = kwargs['x_names']       # Single x variable name
    y_name = kwargs['y_names']       # Single y variable name
    output_file = kwargs['output_file']
    style = kwargs['styles']
    color = kwargs['colors']
    label = kwargs['labels']
    x_log = kwargs['x_log']
    y_log = kwargs['y_log']
    x_label = kwargs['x_label']
    y_label = kwargs['y_label']

    # Verify input
    if data_file == '':
        raise RuntimeError('Data file must be nonempty')
    if x_name == '':
        raise RuntimeError('x_name must be nonempty')
    if y_name == '':
        raise RuntimeError('y_name must be nonempty')
    valid_file = (data_file.endswith('.athdf') or data_file.endswith('.hst') or data_file.endswith('.tab'))
    if not valid_file:
        raise RuntimeError('File must have .athdf, .hst, or .tab extension')

    if style == '':
        style = '-'
    if color == '':
        color = None
    if label == '':
        label = None
    labels_used = False
    if label is not None:
        labels_used = True

    # Load Python plotting modules
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    skip_points=50

    # Read data
    if data_file.endswith('.athdf'):
        data = athena_read.athdf(data_file)
    elif data_file.endswith('.hst'):
        data = athena_read.hst(data_file)
    else:
        data = athena_read.tab(data_file)

    # Extract x and y values
    x_vals = data[x_name].flatten()[::skip_points]
    y_vals = data[y_name].flatten()[::skip_points]

    # Extract basic coordinate information
    if kwargs['direction'] == 1:
        xf = data['x2f']
        xv = data['x2v']
        yf = data['x3f']
        yv = data['x3v']
        zf = data['x1f']

    elif kwargs['direction'] == 2:
        xf = data['x1f']
        xv = data['x1v']
        yf = data['x3f']
        yv = data['x3v']
        zf = data['x2f']

    if kwargs['direction'] == 3:
        xf = data['x1f']
        xv = data['x1v']
        yf = data['x2f']
        yv = data['x2v']
        zf = data['x3f']


    x_min = kwargs['x_min'] if kwargs['x_min'] is not None else xf[0]
    x_max = kwargs['x_max'] if kwargs['x_max'] is not None else xf[-1]
    y_min = kwargs['y_min'] if kwargs['y_min'] is not None else yf[0]
    y_max = kwargs['y_max'] if kwargs['y_max'] is not None else yf[-1]

    x_slice = data[x_name] #each slice is a 3d array where the first side is of size 1, you need to apply the x and y mask for each slice
    y_slice = data[y_name]

    # Create separate masks for x and y
    x_mask = (data['x1v'] >= x_min) & (data['x1v'] <= x_max)
    y_mask = (data['x2v'] >= y_min) & (data['x2v'] <= y_max)
    z_mask = [True]
    # Apply combined mask
    combined_mask = np.zeros(x_slice.shape, dtype=bool)
    combined_mask[0, :, :] = np.outer(y_mask, x_mask)

    x_vals_pdf=x_slice[combined_mask]
    y_vals_pdf=y_slice[combined_mask]

    # print(x_slice)
    # print(x_slice.shape)
    # print(y_slice)
    # print(y_slice.shape)
    # print(x_mask)
    # print(x_mask.shape)
    # print(y_mask)
    # print(y_mask.shape)
    # print(len(data['x1v']))
    # print(len(data['x2v']))

    # Plot data
    plt.figure()
    num_bins = 100
    joint_pdf, xedges, yedges = np.histogram2d(x_vals_pdf, y_vals_pdf, bins=num_bins, density=True)
    joint_pdf=np.log(joint_pdf)
    plt.imshow(joint_pdf.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', cmap='plasma')
    plt.colorbar(label='Probability Density')

    #plt.xlim((x_min, x_max))
    #plt.ylim((y_min, y_max))
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if label is not None:
        plt.legend(loc='best')
    if output_file == 'show':
        plt.show()
    else:
        plt.savefig(output_file+'_joint_pdf.png', bbox_inches='tight')

    # Plot data
    plt.figure()
    plt.plot(x_vals_pdf[::skip_points], y_vals_pdf[::skip_points], style, color=color, label=label)
    #plt.plot(x_vals_pdf, y_vals_pdf, style, color=color, label=label)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    #plt.xlim((x_min, x_max))
    #plt.ylim((y_min, y_max))
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if labels_used:
        plt.legend(loc='best')
    if output_file == 'show':
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')
      
# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      'data_files',
      help='Input file (.athdf, .hst, or .tab)'
    )
    parser.add_argument(
        'x_names',
        help='Name of the abscissa (x-axis variable)'
    )
    parser.add_argument(
        'y_names',
        help='Name of the ordinate (y-axis variable)'
    )
    parser.add_argument(
        'output_file',
        help='Name of output file; use "show" for interactive plot'
    )
    parser.add_argument(
      '-s', '--styles',
      default='-',
      help='Line or marker style (e.g., "-" or "o"); default is solid line'
    )
    parser.add_argument(
      '-c', '--colors',
      default='',
      help='Color code (e.g., "k", "blue", or "#123abc"); default is auto'
    )
    parser.add_argument(
      '-l', '--labels',
      default='',
      help='Label for legend (optional)'
    )
    parser.add_argument(
        '--x_log',
        action='store_true',
        help='Flag to use logarithmic scale for x-axis'
    )
    parser.add_argument(
        '--y_log',
        action='store_true',
        help='Flag to use logarithmic scale for y-axis'
    )
    parser.add_argument('--x_max', type=float, help='Maximum x value')
    parser.add_argument('--x_min', type=float, help='Minimum x value')
    parser.add_argument('--y_min', type=float, help='Minimum y value')
    parser.add_argument('--y_max', type=float, help='Maximum y value')
    parser.add_argument('--x_label', help='Label for x-axis')
    parser.add_argument('--y_label', help='Label for y-axis')
    parser.add_argument('-d', '--direction',
                        type=int,
                        choices=(1, 2, 3),
                        default=3,
                        help=('direction orthogonal to slice for 3D data'))
    
    args = parser.parse_args()
    main(**vars(args))

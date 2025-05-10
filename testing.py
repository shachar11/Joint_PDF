# Data
import numpy as np

# Simplified example data (no random values, just fixed numbers)
data = {
    "Pressure": np.array([[[1013250, 1013250, 1013250, 1013250, 1013250],
                          [1013250, 1013250, 1013250, 1013250, 1013250],
                          [1013250, 1013250, 1013250, 1013250, 1013250],
                          [1013250, 1013250, 1013250, 1013250, 1013250],
                          [1013250, 1013250, 1013250, 1013250, 1013250]]], dtype=np.float32),

    "Temperature": np.array([[[301, 302, 303, 304, 305],
                              [306, 307, 308, 309, 310],
                              [311, 312, 313, 314, 315],
                              [316, 317, 318, 319, 320],
                              [321, 322, 323, 324, 325]]], dtype=np.float32),

    "Fuel-MassFraction": np.array([[[0.1, 0,   0.3, 0.4,  0],
                                    [0  , 0.2, 0.3,   0,  0],
                                    [0.1, 0,    0 ,   0,  0.5],
                                    [0.1, 0.2,   0.3, 0.4,  0.5],
                                    [0  , 0,   0.3,   0,  0.5]]], dtype=np.float32),

    "Density": np.array([[[1001, 1002, 1003, 1004, 1005],
                          [1006, 1007, 1008, 1009, 1010],
                          [1011, 1012, 1013, 1014, 1015],
                          [1016, 1017, 1018, 1019, 1020],
                          [1021, 1022, 1023, 1024, 10025]]], dtype=np.float32)
}

# Skip every 2nd point
skip_points = 2

print(data)
print("-"*50)
# Step 2: Create a mask where Fuel-MassFraction is not zero
mask = data["Fuel-MassFraction"] != 0  # This is a 3D mask (same shape as Fuel-MassFraction)
print(mask)
print("-"*50)
# Step 3: Apply the mask to filter out all lists with corresponding zero values in Fuel-MassFraction
data_filtered_MassFracion_not_0 = {
    key: val[mask]  # Filter using the mask on all keys
    for key, val in data.items()
}
print(data_filtered_MassFracion_not_0)
print("-"*50)
x_vals=[]
x_vals.append(data_filtered_MassFracion_not_0["Temperature"].flatten())
print(x_vals)
print("-"*50)
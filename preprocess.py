import os
from osgeo import gdal
import numpy as np
import sys
import traceback  # Import the traceback module
import matplotlib.pyplot as plt

# Ensure GDAL uses exceptions
gdal.UseExceptions()

# Define the path to your image
file_path = "data/Track1/train/images/0.tif"

# Check if the file exists to avoid Nonetype errors
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

try:
    # Open the image
    ds = gdal.Open(file_path)
    if ds is None:
        print("Failed to open the image")
        sys.exit(1)


except Exception as e:
    print("An error occurred:")
    traceback.print_exc()
    sys.exit(1)


# 6 bands, 16-bit depth

# Get the first raster band
band = ds.GetRasterBand(1)

image = band.ReadAsArray()
print("Image size:", image.shape)

print(ds.GetMetadata())

for i in range(1, ds.RasterCount + 1):
    band = ds.GetRasterBand(i)
    print(f"Band {i} metadata: {band.GetMetadata()}")

for i in range(1, ds.RasterCount + 1):
    band = ds.GetRasterBand(i)
    array = band.ReadAsArray()

    plt.figure()
    plt.imshow(array, cmap="gray")  # Change colormap as needed
    plt.title(f"Band {i}")
    plt.colorbar()
    plt.show()

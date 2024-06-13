import h5py
import numpy as np


file = './data/temp/gzDesi_galaxy_zoo_0-65.h5'
output_file = './data/gzDesi_galaxy_zoo_0-65_filtered_4-5-7-8.h5'

#Classes to extract
filter = [4, 5, 7, 8]

with h5py.File(file, 'r') as f:
    classes = f['classes'][:]
    images = f['images'][:]
    
    filtered_indices = np.where(np.isin(classes, filter))[0]
    
    filtered_classes = classes[filtered_indices]
    filtered_images = images[filtered_indices]
    
    with h5py.File(output_file, 'w') as f_out:
        f_out.create_dataset('classes', data=filtered_classes)
        f_out.create_dataset('images', data=filtered_images)

print(f"Die Zeilen mit dem Wert {filter} bei 'classes' wurden in {output_file} gespeichert.")
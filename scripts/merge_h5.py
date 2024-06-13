import h5py
import numpy as np

file1 = './data/gzDecals_galaxy_zoo_0-75.h5'
file2 = './data/gzDesi_galaxy_zoo_0-65_filtered_4-5-7-8.h5'
output_file = './data/gzDecals-gzDesi_galaxy_zoo_0-75.h5'

with h5py.File(file1, 'r') as f1:
    with h5py.File(file2, 'r') as f2:
        classes1 = f1['classes'][:]
        images1 = f1['images'][:]
        classes2 = f2['classes'][:]
        images2 = f2['images'][:]
        
        combined_classes = np.concatenate((classes1, classes2), axis=0)
        combined_images = np.concatenate((images1, images2), axis=0)
        
        with h5py.File(output_file, 'w') as f_out:
            f_out.create_dataset('classes', data=combined_classes)
            f_out.create_dataset('images', data=combined_images)

print(f"Die Dateien {file1} und {file2} wurden erfolgreich vereint und in {output_file} gespeichert.")
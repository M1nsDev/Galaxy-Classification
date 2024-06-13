import os
import glob
import pyarrow.parquet as pq

parquet_folder = "./data/parquet"
output_folder = "./data/images/gz_desi"

os.makedirs(output_folder, exist_ok=True)

parquet_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))

def process_parquet_file(parquet_file, batch_size=15000):
    print(f"Processing {parquet_file}")

    try:
        parquet_file_obj = pq.ParquetFile(parquet_file)
        base_name = os.path.splitext(os.path.basename(parquet_file))[0]
        subfolder_path = os.path.join(output_folder, base_name)
        os.makedirs(subfolder_path, exist_ok=True)

        for batch in parquet_file_obj.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            for index, row in df.iterrows():
                image_bytes = row["image"]["bytes"]
                image_id = row["dr8_id"]

                file_name = f"{image_id}.png"
                file_path = os.path.join(subfolder_path, file_name)

                with open(file_path, "wb") as image_file:
                    image_file.write(image_bytes)

                print(f"Saved {file_path}")

    except Exception as e:
        print(f"Error processing {parquet_file}: {e}")

for parquet_file in parquet_files:
    process_parquet_file(parquet_file)

print("All files processed.")
import pandas as pd
from services.route_optimizer_new import generate_route_map
import shutil
import os
# import asyncio

async def compress_directory(directory_path, zip_file_path):
    try:
        if os.path.exists(zip_file_path):
            shutil.rmtree(zip_file_path)
        else:
            shutil.make_archive(zip_file_path, 'zip', directory_path)
            print(f"Directory {directory_path} has been compressed into {zip_file_path}.zip")
    except Exception as e:
        print(f"An error occurred: {e}")

async def copy_directory(source_dir, dest_dir):
    try:
        # Copy the directory and all its contents
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
        print(f"Directory {source_dir} has been copied to {dest_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")

async def make_data_for_download():
    dataframe = pd.read_csv("Benefits_new_data.csv")
    dataframe.insert(1, "Driving Time (min) Optimal", "")
    dataframe.insert(2, "Driving Distance (mile) Optimal", "")
    dataframe.insert(3, "Driving Time (min) Manual", "")
    dataframe.insert(4, "Driving Distance (mile) Manual", "")
    dataframe.insert(5, "Percentage of DRT", "")
    dataframe.insert(6, "Percentage of Swing", "")
    dataframe.insert(7, "Number of Stops", "")
    dataframe.insert(8, "Route Optimal", "")
    dataframe.insert(9, "Route Manual", "")
    dataframe.insert(10, "DATE", "")
    dataframe.insert(11, "HF_DIVISION_NAME", "")
    dataframe.insert(12, "HF_SITE_NAME", "")
    dataframe.insert(13, "HF_ADDRESS_LINE1", "")
    dataframe.insert(14, "HF_ADDRESS_LINE2", "")
    dataframe.insert(15, "HF_ADDRESS_CITY", "")
    dataframe.insert(16, "HF_ADDRESS_STATE", "")
    dataframe.insert(17, "HF_ADDRESS_POSTAL_CODE", "")
    dataframe.insert(18, "DF_FACILITY_NAME", "")
    dataframe.insert(19, "DF_ADDRESS_LINE1", "")
    dataframe.insert(20, "DF_ADDRESS_LINE2", "")
    dataframe.insert(21, "DF_ADDRESS_CITY", "")
    dataframe.insert(22, "DF_ADDRESS_STATE", "")
    dataframe.insert(23, "DF_ADDRESS_POSTAL_CODE", "")
    dataframe.insert(24, "Time Benefit", "")
    dataframe.insert(25, "Distance Benefit", "")
    dataframe.insert(26, "Benefit", "")
    location_ids = dataframe["Route_ID"]
    os.makedirs("services/route_optimization_output/Sequence", exist_ok=True)
    os.makedirs("services/route_optimization_output/IND_results", exist_ok=True)
    for location_id in location_ids:
        data = await generate_route_map(location_id)
        # print(data)
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        data = data[dataframe.columns]
        row_to_assign = data.iloc[0:1]
        dataframe.loc[dataframe['Route_ID'] == location_id, :] = row_to_assign.values[0]
    dataframe.to_csv("services/route_optimization_output/Updated_Benefits_data.csv", index=False)
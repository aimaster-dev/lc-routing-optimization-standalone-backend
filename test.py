from arcgis.gis import GIS
from arcgis.network import analysis
from arcgis.features import FeatureSet

def get_distance_time_matrix_arcgis(locations):
    target_length = len(locations)
    gis = GIS("https://www.arcgis.com", "Bhairavmehta", "Repsrv2025#")

    location_dict = {"features": [], "geometryType": "esriGeometryPoint"}
    destination_number = len(locations)

    for location in locations:
        # location = (lon, lat)
        location_feature = {
            "geometry": {"x": location[0], "y": location[1]},  # lon, lat
            "attributes": {}
        }
        location_dict["features"].append(location_feature)

    fs_locations = FeatureSet.from_dict(location_dict)

    result = analysis.generate_origin_destination_cost_matrix(
        origins=fs_locations,
        destinations=fs_locations,
        travel_mode="Driving Time",
        number_of_destinations_to_find=destination_number
    )

    if result.solve_succeeded:
        od_matrix_df = result.output_origin_destination_lines.sdf
        distance_matrix = [[0] * destination_number for _ in range(destination_number)]
        time_matrix = [[0] * destination_number for _ in range(destination_number)]

        for _, row in od_matrix_df.iterrows():
            origin_index = int(row["OriginOID"])
            destination_index = int(row["DestinationOID"])
            distance_matrix[origin_index-1][destination_index-1] = row.get("Total_Distance", 0)
            time_matrix[origin_index-1][destination_index-1] = row["Total_Time"]

        return distance_matrix, time_matrix
    else:
        print("❌ Failed to generate OD cost matrix.")
        print("Message:", result.messages)
        return None, None


# === Run Test ===
locations = [(-118.25, 34.05), (-118.24, 34.06)]  # LA area
distance_matrix, time_matrix = get_distance_time_matrix_arcgis(locations)

print("✅ Distance Matrix:")
print(distance_matrix)

print("\n✅ Time Matrix:")
print(time_matrix)

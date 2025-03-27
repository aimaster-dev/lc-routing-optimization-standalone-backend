import pandas as pd

file_path = 'uploaded_files\\transformed_data_snowflk.csv'

data = pd.read_csv(file_path)

print(data.head())

route_number = "3327_2024-11-20"
filtered_data = data[data['Route #'].astype(str) == route_number]

# Save the filtered data to a new CSV file
filtered_data_path = f'filtered_data_route_3327_2024-11-20.csv'
filtered_data.to_csv(filtered_data_path, index=False)
# import re

# route_manual_data = [['Haul', 'Stop 1', 'LF1', 'Haul', 'Stop 2', 'LF1', 'Stop 2', 'Stop 3', 'LF1', 'Stop 3', 'Stop 4', 'LF1', 'Stop 4', 'Stop 5', 'LF1', 'Stop 5', 'Stop 6', 'LF1', 'Stop 6', 'Stop 7', 'LF1', 'Stop 7', 'Stop 8', 'LF1', 'Stop 8', 'Haul', 'Stop 9', 'LF1', 'Haul', 'Stop 10', 'LF1', 'Stop 10', 'Haul'], ['Haul', 'SWG', 'LF1', 'Haul', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'Haul', 'SWG', 'LF1', 'Haul', 'DRT', 'LF1', 'DRT', 'Haul']]
# route_optimal_data = [['Haul', 'Stop 4', 'LF1', 'Stop 4', 'Stop 10', 'LF1', 'Stop 10', 'Stop 8', 'LF1', 'Stop 8', 'Stop 7', 'LF1', 'Stop 7', 'Stop 3', 'LF1', 'Stop 3', 'Stop 2', 'LF1', 'Stop 2', 'Stop 5', 'LF1', 'Stop 5', 'Stop 6', 'LF1', 'Stop 6', 'Haul', 'Stop 1', 'LF1', 'Stop 9', 'LF1', 'Haul'], ['Haul', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'DRT', 'LF1', 'DRT', 'Haul']]

# table_manual_data = []
# stop_manual_data = []
# operation_manual_data = []
# index_manual = []

# past_index_match = 0
# stop_length = 1
# for index, node in enumerate(route_manual_data[0]):
#     if index == 0:
#         continue
#     index_match = re.match(r'Stop (\d+)', node)
#     if index_match:
#         index_match_num = index_match.group(1)
#         if index_match_num == past_index_match:
#             stop_length += 1
#             continue
#         else:
#             stop_manual_data.append(index_match_num)
#             operation_manual_data.append(route_manual_data[1][index])
#             index_manual.append(index)
#             stop_length = 1
#             past_index_match = index_match_num
#     else:
#         stop_length += 1

# table_manual_data.append(stop_manual_data)
# table_manual_data.append(operation_manual_data)
# table_manual_data.append(index_manual)

# print(len(route_optimal_data[0]))
# print(len(route_optimal_data[1]))

# table_optimal_data = []
# stop_optimal_data = []
# operation_optimal_data = []
# index_optimal = []

# past_index_match = 0
# stop_length = 1
# for index, node in enumerate(route_optimal_data[0]):
#     if index == 0:
#         continue
#     index_match = re.match(r'Stop (\d+)', node)
#     if index_match:
#         index_match_num = index_match.group(1)
#         if index_match_num == past_index_match:
#             stop_length += 1
#             continue
#         else:
#             stop_optimal_data.append(index_match_num)
#             operation_optimal_data.append(operation_manual_data[int(index_match_num) - 1])
#             index_optimal.append(index)
#             stop_length = 1
#             past_index_match = index_match_num
#     else:
#         stop_length += 1

# table_optimal_data.append(stop_optimal_data)
# table_optimal_data.append(operation_optimal_data)
# table_optimal_data.append(index_optimal)

# print(table_manual_data)
# print(table_optimal_data)

# # stop_length = 2
# # past_index_match = re.match(r'Stop (\d+)', route_manual_data[0][1]).group(1)
# # stop_manual_data.append(f"Stop {past_index_match}")
# # operation_manual_data.append(route_manual_data[1][1])
# # for index, node in enumerate(route_manual_data[0]):
# #     print(f"index {len(route_manual_data[0])}: ", index)
# #     if index == len(route_manual_data[0]) - 1:
# #         # print(f"index {len(route_manual_data[0])}: ", index)
# #         length_manual.append(stop_length)
# #         # operation_manual_data.append()
# #     index_match = re.match(r'Stop (\d+)', node)
# #     if index_match:
# #         index_match_num = index_match.group(1)
# #         if index_match_num == past_index_match:
# #             stop_length += 1
# #             continue
# #         else:
# #             stop_manual_data.append(f"Stop {index_match.group(1)}")
# #             operation_manual_data.append(route_manual_data[1][index])
# #             length_manual.append(stop_length - 2)
# #             stop_length = 2
# #             past_index_match = index_match.group(1)
# #     else:
# #         stop_length += 1
# #         continue
# # table_manual_data.append(stop_manual_data)
# # table_manual_data.append(operation_manual_data)
# # table_manual_data.append(length_manual)

# # print(table_manual_data)
import os
from pathlib import Path
import json

file_path = Path(r"data\cord_subset\images\cord_0011.json")

with open(file_path, 'r', encoding='utf-8') as f:
    file_json = json.load(f)
first_dict = file_json[0]
main_keys = first_dict.keys() 
if 'doc_preprocessor_res' in first_dict and 'angle' in first_dict['doc_preprocessor_res']:
    print(f"angle: {first_dict['doc_preprocessor_res']['angle']}")
# for main_key in main_keys:
#     print(f'{main_key}')
#     if isinstance(first_dict[main_key], dict):
#         print(f"--- {first_dict[main_key].keys()}")
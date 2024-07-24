from segmentation_definitions_original import OBJECT_STR_TO_DESCR  as str2d
from constants import ALL_DETECTOR as all
from segmentation_definitions_original import _RECEPTACLE_OBJECTS as receps_original
from segmentation_definitions import OBJECT_CLASSES as oc
a = list()

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

duplicates = find_duplicates(oc)

if duplicates:
    print("중복된 요소가 발견되었습니다:")
    for item in duplicates:
        print(item)
else:
    print("중복된 요소가 없습니다.")






#################
# for i in all:
#     if i not in list(str2d.keys()):
#         if sum(1 for c in i if c.isupper()) >= 2:
#             a.append(i)
            
# def create_dict_with_changes(lst):
#     result = {}
    
#     for item in lst:
#         updated_item = ''
#         previous_char_is_upper = False
        
#         for char in item:
#             if char.isupper():
#                 if previous_char_is_upper:
#                     updated_item += char.lower()
#                 else:
#                     updated_item += ' ' + char.lower()
#                 previous_char_is_upper = True
#             else:
#                 updated_item += char
#                 previous_char_is_upper = False
        
#         result[item] = updated_item.strip()
    
#     return result

# b = create_dict_with_changes(a)
# for k, v in b.items():
#     print(f"'{k}': '{v}'")
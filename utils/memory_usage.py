import sys
import pandas as pd
import numpy as np


def get_size(obj):
    """Get size of object in bytes"""
    size = sys.getsizeof(obj)
    if isinstance(obj, pd.DataFrame):
        size += obj.memory_usage(deep=True).sum()
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    return size

def see_memory_usage(global_items):
    # Get all variables from globals()
    variables = [(var_name, var_obj) for var_name, var_obj in global_items
                 if not var_name.startswith('_') and not callable(var_obj)]

    # Create list of tuples with variable info
    var_info = []
    for var_name, var_obj in variables:
        size_bytes = get_size(var_obj)
        var_type = type(var_obj).__name__

        # Convert to human readable format
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            size_str = f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 ** 3:
            size_str = f"{size_bytes / 1024 ** 2:.2f} MB"
        else:
            size_str = f"{size_bytes / 1024 ** 3:.2f} GB"

        var_info.append({
            'Variable': var_name,
            'Type': var_type,
            'Size': size_str,
            'Size_Bytes': size_bytes
        })

    # Create DataFrame and sort by size
    memory_usage_df = pd.DataFrame(var_info)
    memory_usage_df = memory_usage_df.sort_values('Size_Bytes', ascending=False)
    memory_usage_df = memory_usage_df.drop('Size_Bytes', axis=1).reset_index(drop=True)

    return memory_usage_df
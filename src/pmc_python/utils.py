import pandas as pd
from typing import List, Union, Dict, Any, Optional


def path_string(names: List[str], levels: List[int]) -> str:
    """
    Print a hierarchical path string from a vector of names and levels
    
    Parameters
    ----------
    names : List[str]
        A list of names
    levels : List[int]
        A list of numbers with indentation level
        
    Returns
    -------
    str
        Hierarchical path string
        
    Examples
    --------
    >>> names = ["carnivores", "bears", "polar", "grizzly", "cats", "tiger", "rodents"]
    >>> levels = [1, 2, 3, 3, 2, 3, 1]
    >>> path_string(names, levels)
    """
    if len(names) != len(levels):
        raise ValueError("names and levels should be the same length")
    
    if not all(isinstance(n, (int, float)) for n in levels):
        raise ValueError("levels should be a list of numbers")
    
    if not names:
        return ""
    
    min_level = min(levels)
    if min_level > 1:
        levels = [n - min_level + 1 for n in levels]
    
    paths = []
    current_path = [""] * (max(levels) + 1)  # Add buffer for safety
    
    for i, (name, level) in enumerate(zip(names, levels)):
        while len(current_path) < level:
            current_path.append("")
        
        current_path[level - 1] = name
        current_path = current_path[:level]
        
        path_str = "; ".join(current_path)
        path_str = path_str.replace("NA; ", "").replace("; NA", "")
        paths.append(path_str)
    
    return paths[-1] if paths else ""


def repeat_sub(df: pd.DataFrame, column: str = "subheading", first: bool = True) -> pd.DataFrame:
    """
    Repeat table subheadings in a new column
    
    Identifies subheadings in a DataFrame by checking for rows with a non-empty
    first column and all other columns are empty. Removes subheader rows and
    repeats values down a new column.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with subheadings
    column : str, default "subheading"
        New column name
    first : bool, default True
        Add subheader as first column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with subheadings repeated
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'genes': ['Up', 'aroB', 'glnP', 'Down', 'ndhA', 'pyrF'],
    ...     'fold_change': [None, 2.5, 1.7, None, -3.1, -2.6]
    ... })
    >>> repeat_sub(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df should be a DataFrame")
    
    if df.shape[1] == 1:
        return df
    
    other_cols = df.iloc[:, 1:]
    is_subheader = other_cols.isna().all(axis=1) | (other_cols == "").all(axis=1) | (other_cols == "\u00A0").all(axis=1)
    
    if not is_subheader.any():
        return df
    
    subheader_indices = is_subheader[is_subheader].index.tolist()
    
    consecutive_count = sum(1 for i in range(len(subheader_indices) - 1) 
                           if subheader_indices[i+1] - subheader_indices[i] == 1)
    
    if consecutive_count > 1:
        return df
    
    if subheader_indices[0] != 0:
        return df
    
    result_df = df.copy()
    
    subheader_values = []
    for i in range(len(df)):
        if i in subheader_indices:
            current_subheader = df.iloc[i, 0]
        else:
            if i > 0:
                prev_subheader_idx = max([idx for idx in subheader_indices if idx < i], default=0)
                current_subheader = df.iloc[prev_subheader_idx, 0]
            else:
                current_subheader = None
        subheader_values.append(current_subheader)
    
    result_df[column] = subheader_values
    
    result_df = result_df[~is_subheader]
    
    result_df = result_df.infer_objects()
    
    if first:
        cols = [column] + [col for col in result_df.columns if col != column]
        result_df = result_df[cols]
    
    return result_df.reset_index(drop=True)


def collapse_rows(tables: Union[List[pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame], 
                  na_string: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Collapse a list of PubMed Central tables
    
    Collapse rows into a semi-colon delimited list with column names and cell values
    
    Parameters
    ----------
    tables : Union[List[pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]
        A list or dict of tables, usually from pmc_table
    na_string : str, optional
        Additional cell values to skip, default is None
        
    Returns
    -------
    pd.DataFrame or None
        DataFrame with table and row number and collapsed text
        
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'genes': ['aroB', 'glnP', 'ndhA', 'pyrF'],
    ...     'fold_change': [2.5, 1.7, -3.1, -2.6]
    ... })
    >>> collapse_rows({'Table 1': df})
    """
    if tables is None:
        return None
    
    if isinstance(tables, pd.DataFrame):
        tables = {'Table': tables}
    elif isinstance(tables, list):
        tables = {f'Table {i+1}': table for i, table in enumerate(tables)}
    
    if not isinstance(tables, dict):
        raise ValueError("tables should be a list or dict of DataFrames from pmc_table")
    
    if not tables or not isinstance(list(tables.values())[0], pd.DataFrame):
        raise ValueError("tables should be a list or dict of DataFrames from pmc_table")
    
    all_results = []
    
    for table_name, df in tables.items():
        if df.empty:
            continue
        
        df_copy = df.copy()
        
        for col in df_copy.columns:
            if df_copy[col].dtype == 'category':
                df_copy[col] = df_copy[col].astype(str)
        
        collapsed_rows = []
        
        for idx, row in df_copy.iterrows():
            row_parts = []
            
            for col_name, value in row.items():
                if pd.isna(value) or str(value) == "" or str(value) == "\u00A0":
                    continue
                if na_string is not None and str(value) == na_string:
                    continue
                
                row_parts.append(f"{col_name}={value}")
            
            if row_parts:
                collapsed_rows.append({
                    'table': table_name,
                    'row': idx + 1,
                    'text': "; ".join(row_parts)
                })
        
        all_results.extend(collapsed_rows)
    
    if not all_results:
        return None
    
    return pd.DataFrame(all_results)

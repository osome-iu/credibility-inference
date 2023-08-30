"""
Label users. 
Args: test_size (optional): Hide a portion of domains. 
-> known users
-> test users
Input: 
    - in_fpath (.parquet file name): df of posts, with a column named 'domain'
    - domain_label_fpath (.csv file name): df of domain labels
Output: 
    - out_fpath (.parquet file name): df of users
"""

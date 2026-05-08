import random
import numpy as np
import pandas as pd
from datetime import datetime
import os
import shutil
import sys

# Extend path so we can import src/config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import DOMAIN, n_random_item, num_users_to_sample, DATASET_ROOT

random_seed = 23
random.seed(random_seed)
np.random.seed(random_seed)

def data_prepare():
    domain = DOMAIN  
    print(f"Processing single domain: {domain} with {num_users_to_sample} users")

    inter_path = str(DATASET_ROOT / "filtered_data" / "inter" / f"inter_{domain}.csv")
    meta_path = str(DATASET_ROOT / "filtered_data" / "meta" / f"meta_{domain}.csv")

    print(f"Loading inter data from: {inter_path}")
    inter_df = pd.read_csv(inter_path)
    print(f"Loading meta data from: {meta_path}")
    meta_df = pd.read_csv(meta_path)

    # 3. Filter users
    # rating >= 4
    inter_df = inter_df[inter_df['rating'] >= 4]
    
    # Count interactions per user
    user_counts = inter_df.groupby("user_id").size().reset_index(name='counts')
    # Keep users whose interaction count is between 10 and 100
    valid_users = user_counts[(user_counts['counts'] >= 10) & (user_counts['counts'] <= 100)]['user_id'].tolist()
    
    # Sample the specified number of users
    if len(valid_users) < num_users_to_sample:
        print(f"Warning: Only {len(valid_users)} valid users found, which is less than requested {num_users_to_sample}. Using all of them.")
        selected_users = valid_users
    else:
        selected_users = random.sample(valid_users, num_users_to_sample)
    
    selected_inter_df = inter_df[inter_df['user_id'].isin(selected_users)]
    selected_items = selected_inter_df['parent_asin'].unique().tolist()
    
    print(f"Selected {len(selected_users)} users and {len(selected_items)} items.")
    print(f"Total interactions: {len(selected_inter_df)}")

    # 5. Prepare time-series data and split into train/test sets
    selected_inter_df = selected_inter_df.sort_values(by=['user_id', 'timestamp'])
    
    # Split logic: use each user's last interaction as test, the rest as train
    test_df = selected_inter_df.groupby('user_id').tail(1)
    train_df = selected_inter_df.drop(test_df.index)

    # 6. Save data (append the user-count suffix)
    base_dir = str(DATASET_ROOT / "user_item_data" / f"{domain}_{num_users_to_sample}")
    os.makedirs(f"{base_dir}/timesequence", exist_ok=True)
    os.makedirs(f"{base_dir}/random", exist_ok=True)
    
    train_df.to_csv(f"{base_dir}/timesequence/inter_timesequence_train.csv", index=False)
    test_df.to_csv(f"{base_dir}/timesequence/inter_timesequence_test.csv", index=False)
    selected_inter_df.to_csv(f"{base_dir}/timesequence/inter_timesequence_all.csv", index=False)
    
    # Save meta data (only items interacted by the selected users)
    final_meta_df = meta_df[meta_df['parent_asin'].isin(selected_items)]
    final_meta_df.to_csv(f"{base_dir}/meta.csv", index=False)

    # 7. Initialize Memory (User & Item Descriptions) (append the user-count suffix)
    init_dir = str(DATASET_ROOT / "initial" / f"{domain}_{num_users_to_sample}")
    os.makedirs(f"{init_dir}/item", exist_ok=True)
    os.makedirs(f"{init_dir}/user", exist_ok=True)
    
    # Item Memory
    for _, row in final_meta_df.iterrows():
        itemId = row['parent_asin']
        item_info = {
            'main_category': row.get('main_category', ''),
            'item_title': row.get('title', ''),
            'item_subtitle': row.get('subtitle', ''),
            'item_class': row.get('categories', ''),
            'item_price': row.get('price', '')
        }
        mem_str = ", ".join([f"'{k}': '{v}'" for k, v in item_info.items()])
        with open(f"{init_dir}/item/item.{itemId}", "w", encoding="utf-8") as f:
            f.write(mem_str)
            
    # User Memory
    for user in selected_users:
        mem_str = f"I enjoy {domain} very much."
        with open(f"{init_dir}/user/user.{user}", "w", encoding="utf-8") as f:
            f.write(mem_str)
    
    # User Long Memory
    if os.path.exists(f"{init_dir}/user-long"):
        shutil.rmtree(f"{init_dir}/user-long")
    shutil.copytree(f"{init_dir}/user", f"{init_dir}/user-long")

    print(f"Data preparation for {domain} with {num_users_to_sample} users completed successfully.")
    print(f"Outputs stored in {base_dir} and {init_dir}")

if __name__ == "__main__":
    data_prepare()

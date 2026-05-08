
import os
import sys
import pickle
import random

import pandas as pd

# Make src/config.py importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sasrec.model import SASREC
from sasrec.sampler import WarpSampler
from sasrec.util import SASRecDataSet
from config import (
    DOMAIN,
    item_data_source,
    candidate_num,
    inter_data_source,
    num_users_to_sample,
    BASELINES_ROOT,
    createItemDF,
)

if __name__ == "__main__":
    domain = DOMAIN
    print(f"Processing single domain: {domain}")

    # Load interaction data
    inter_all_path = inter_data_source("all")
    inter_all_DF = pd.read_csv(inter_all_path, encoding="utf-8", dtype=str)
    
    # Unify IDs as strings
    inter_all_DF["user_id"] = inter_all_DF["user_id"].astype(str)
    inter_all_DF["parent_asin"] = inter_all_DF["parent_asin"].astype(str)

    inter_processed_DF = (inter_all_DF.rename(columns={'user_id': 'userID', 'parent_asin': 'itemID', 'timestamp': 'time'})
                    .sort_values(by=['userID', 'time'])
                    .drop(['rating', 'time'], axis=1)
                    .reset_index(drop=True)[["userID", "itemID"]])

    user_set, item_set = set(inter_processed_DF['userID'].unique()), set(inter_processed_DF['itemID'].unique())
    user_map = {user: idx + 1 for idx, user in enumerate(user_set)}
    item_map = {item: idx + 1 for idx, item in enumerate(item_set)}

    inter_processed_DF["userID"] = inter_processed_DF["userID"].map(user_map)
    inter_processed_DF["itemID"] = inter_processed_DF["itemID"].map(item_map)

    # Save SASRec data
    save_dir = str(BASELINES_ROOT / "SASRec" / f"{domain}_{num_users_to_sample}")
    os.makedirs(save_dir, exist_ok=True)
    inter_processed_DF.to_csv(os.path.join(save_dir, 'sasrec_data.txt'), sep="\t", header=False, index=False)
    
    # Save maps
    with open(os.path.join(save_dir, 'maps.pkl'), 'wb') as f:
        pickle.dump((user_map, item_map), f)

    # Prepare the dataset
    data = SASRecDataSet(os.path.join(save_dir, 'sasrec_data.txt'))
    data.split()  # Train, validation, test split

    # Model parameters
    max_len = 80
    hidden_units = 64 # For 27k items, 64 dimensions is a good balance
    batch_size = 128  # For 2000 users, increasing the batch to 128 can significantly speed up training

    model = SASREC(
        item_num=data.itemnum,
        seq_max_len=max_len,
        num_blocks=2, # Increase to 2 layers to improve the ability to capture complex sequential features
        embedding_dim=hidden_units,
        attention_dim=hidden_units,
        attention_num_heads=1,
        dropout_rate=0.4,
        conv_dims=[hidden_units, hidden_units],
        l2_reg=0.00001
    )

    sampler = WarpSampler(data.user_train, data.usernum, data.itemnum, batch_size=batch_size, maxlen=max_len, n_workers=1)
    model.build((None, max_len))
    model.train(
        data,
        sampler,
        num_epochs=50, # For 27k items, running the full 50 epochs is recommended
        batch_size=batch_size,
        lr=0.001,
        val_epoch=2,   # Validate once every 2 epochs
        val_target_user_n=min(200, data.usernum), # Sample 200 users for validation
        target_item_n=100,
        auto_save=True,
        path=save_dir,
        exp_name='exp_example',
    )

    # Load test data
    inter_test_path = inter_data_source("test")
    
    inter_test_DF = pd.read_csv(inter_test_path, encoding="utf-8", dtype=str)
    itemDF = createItemDF(item_data_source)
    itemDF["parent_asin"] = itemDF["parent_asin"].astype(str)
    all_item_ids = [iid for iid in itemDF["parent_asin"].unique().tolist() if iid in item_map]
    

    print(f"Starting evaluation with global random negative sampling (Pool size: {len(all_item_ids)})...")
    for index, record in inter_test_DF.iterrows():
        try:
            target_itemId = str(record["parent_asin"])
            userId = str(record["user_id"])

            if userId not in user_map or target_itemId not in item_map:
                continue

            random_itemId_list = []
            while len(random_itemId_list) < candidate_num - 1:
                n_id = random.choice(all_item_ids)
                if n_id != target_itemId and n_id not in random_itemId_list:
                    random_itemId_list.append(n_id)
            
            random_itemId_list.append(target_itemId)
            random.shuffle(random_itemId_list)

            # Get scores
            score = model.get_user_item_score(data, [userId], random_itemId_list, user_map, item_map, batch_size=1)
            
            # Parse scores and sort
            if len(score) > 0:
                score_row = score.iloc[0]
                scores = []
                for item_id in random_itemId_list:
                    if item_id in score_row:
                        scores.append((item_id, score_row[item_id]))
                    else:
                        scores.append((item_id, 0))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                sorted_item_ids = [item_id for item_id, _ in scores]
                relevance_score_list = [1 if x == target_itemId else 0 for x in sorted_item_ids]

                target_rank = relevance_score_list.index(1) + 1

        except Exception as e:
            print(f"Error at index {index}: {e}")
            continue

    print("Evaluation finished.")
    exit()
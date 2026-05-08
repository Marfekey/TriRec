from prompt import *
import random
import re
from fuzzywuzzy import fuzz
import shutil
import os
from config import (
    model,
    inter_data_source,
    item_data_source,
    DOMAIN,
    num_users_to_sample,
    DATASET_ROOT,
    MEMORY_ROOT,
    createInterDF,
    createItemDF,
)
from request import parallel_get_responses, MAX_WORKERS, API_BATCH
from tqdm import tqdm

mode = "train"
exp_name = f"{DOMAIN}_{num_users_to_sample}"


def initialize_memory(exp_name: str) -> None:
    """Copy the initial user / item / user-long memory from dataset/initial."""
    memory_dir = MEMORY_ROOT / exp_name
    if (memory_dir / "item").exists() or (memory_dir / "user").exists():
        print(f"Memory directory {memory_dir} already exists. Skipping copy.")
        return

    initial_path = DATASET_ROOT / "initial" / f"{DOMAIN}_{num_users_to_sample}"
    shutil.copytree(initial_path / "item", memory_dir / "item")
    shutil.copytree(initial_path / "user", memory_dir / "user")
    shutil.copytree(initial_path / "user-long", memory_dir / "user-long")

def save_memory(ratio: str) -> None:
    src_folder = MEMORY_ROOT / exp_name
    dst_folder = MEMORY_ROOT / f"{exp_name}_{ratio}"
    try:
        shutil.copytree(src_folder, dst_folder)
        print(f"Folder '{src_folder}' successfully copied to '{dst_folder}'")
    except Exception as e:
        print(f"Error copying folder: {e}")

def process_interaction(interDF, itemDF, exp_name, model):
    """Batch-process interactions in a single domain: system choice, user update, and item update are all parallelized."""
    # Force ID to string to prevent 000123 from being treated as 123
    interDF["user_id"] = interDF["user_id"].astype(str)
    interDF["parent_asin"] = interDF["parent_asin"].astype(str)
    itemDF["parent_asin"] = itemDF["parent_asin"].astype(str)
    
    # Get the global item pool
    all_item_ids = itemDF["parent_asin"].unique().tolist()
    
    all_inter_num = interDF.shape[0]
    save_interval = max(1, int(all_inter_num * 0.1))
    processed = 0

    # Process interactions in batches
    for start in tqdm(range(0, all_inter_num, API_BATCH)):
        end = min(start + API_BATCH, all_inter_num)
        batch = interDF.iloc[start:end]

        # Filter and read memory info
        valid_data = []
        for _, row in batch.iterrows():
            u_id = str(row["user_id"])
            p_id = str(row["parent_asin"])
            
            # Global random negative sampling: pick an item from the pool that differs from p_id
            n_id = random.choice(all_item_ids)
            while n_id == p_id:
                n_id = random.choice(all_item_ids)
            
            u_path = str(MEMORY_ROOT / exp_name / "user" / f"user.{u_id}")
            p_path = str(MEMORY_ROOT / exp_name / "item" / f"item.{p_id}")
            n_path = str(MEMORY_ROOT / exp_name / "item" / f"item.{n_id}")
            
            if os.path.exists(u_path) and os.path.exists(p_path) and os.path.exists(n_path):
                try:
                    with open(u_path, "r", encoding="utf-8") as f: u_mem = f.read()
                    with open(p_path, "r", encoding="utf-8") as f: p_mem = f.read()
                    with open(n_path, "r", encoding="utf-8") as f: n_mem = f.read()
                    
                    p_title = itemDF[itemDF["parent_asin"] == p_id]["title"].values[0]
                    n_title = itemDF[itemDF["parent_asin"] == n_id]["title"].values[0]
                    
                    valid_data.append({
                        "u_id": u_id, "p_id": p_id, "n_id": n_id,
                        "u_mem": u_mem, "p_mem": p_mem, "n_mem": n_mem,
                        "p_title": p_title, "n_title": n_title
                    })
                except Exception:
                    continue
            else:
                # For debugging: print which file is missing when data is skipped
                # print(f"Missing: u:{os.path.exists(u_path)} p:{os.path.exists(p_path)} n:{os.path.exists(n_path)} for {u_id}")
                continue
        
        if not valid_data:
            processed += len(batch)
            continue

        # Build system selection prompts
        system_prompts = []
        for d in valid_data:
            # Positive sample first, negative sample second
            list_of_item_description = (
                f"title:{d['p_title']}. description:{d['p_mem'].strip()}\n"
                f"title:{d['n_title'].strip()}. description:{d['n_mem'].strip()}"
            )
            system_prompts.append(system_prompt_template(d['u_mem'], list_of_item_description))

        # Get system choice responses in parallel
        system_responses = parallel_get_responses(system_prompts, model, max_workers=MAX_WORKERS)

        # Parse system choice and explanation
        selected_titles, system_reasons = [], []
        for resp in system_responses:
            if not resp:
                selected_titles.append(""); system_reasons.append("")
                continue
            try:
                sel, reason = parse_response(resp)
                selected_titles.append(sel); system_reasons.append(reason)
            except Exception:
                selected_titles.append(""); system_reasons.append("")

        # Determine whether the choice is correct and build feedback prompts
        user_prompts, item_prompts = [], []
        for i, d in enumerate(valid_data):
            sel_title = selected_titles[i]
            sys_reason = system_reasons[i]
            
            pos_similarity = fuzz.ratio(sel_title.lower(), d['p_title'].lower()) if d['p_title'] else 0
            neg_similarity = fuzz.ratio(sel_title.lower(), d['n_title'].lower()) if d['n_title'] else 0
            is_choice_right = pos_similarity > neg_similarity
            
            # Keep a consistent order: positive sample first, negative sample last
            list_of_item_description = (
                f"title:{d['p_title']}. description:{d['p_mem'].strip()}\n"
                f"title:{d['n_title'].strip()}. description:{d['n_mem'].strip()}"
            )
            u_prompt, i_prompt = create_prompts(d['u_mem'], list_of_item_description, d['p_title'], d['n_title'], sys_reason, is_choice_right)
            user_prompts.append(u_prompt)
            item_prompts.append(i_prompt)

        # Update user and item memory in parallel
        user_update_responses = parallel_get_responses(user_prompts, model, max_workers=MAX_WORKERS)
        item_update_responses = parallel_get_responses(item_prompts, model, max_workers=MAX_WORKERS)

        # Write memory back
        for i, d in enumerate(valid_data):
            try:
                if user_update_responses[i]:
                    update_user_memory(d['u_id'], exp_name, user_update_responses[i])
                if item_update_responses[i]:
                    update_item_memory(d['p_id'], d['n_id'], exp_name, item_update_responses[i])
                print(f"\n{d['u_id']} {d['p_id']} already done.")
            except Exception as e:
                print(f"Error writing memory for {d['u_id']}: {e}")

        # Compute the progress ratio before and after this batch
        old_ratio = (processed) // save_interval
        processed += len(batch)
        new_ratio = processed // save_interval
        
        # Save when a new interval is crossed (e.g. from 0.9 to 1.1)
        if new_ratio > old_ratio:
            # Cap the ratio at 10
            save_ratio = min(10, int(new_ratio))
            save_memory(str(save_ratio))

# Single-domain mode no longer needs cross-domain negative sampling logic

def parse_response(responseText):
    selected_item_title = re.split(r"Choice:|\n", responseText)[1]
    system_reason = re.split(r"Explanation:", responseText)[-1].strip()
    return selected_item_title, system_reason

def create_prompts(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason, is_choice_right):
    if not is_choice_right:
        user_prompt = user_prompt_system_role(user_description) + '\n' + user_prompt_template(list_of_item_description, pos_item_title, neg_item_title, system_reason)
        item_prompt = item_prompt_template(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason)
    else:
        user_prompt = user_prompt_system_role(user_description) + '\n' + user_prompt_template_true(list_of_item_description, pos_item_title, neg_item_title, system_reason)
        item_prompt = item_prompt_template_true(user_description, list_of_item_description, pos_item_title, neg_item_title)
    return user_prompt, item_prompt

def update_user_memory(userId, exp_name, responseText):
    responseText = responseText.split("My updated self-introduction:")[-1].strip()
    user_memory_path = MEMORY_ROOT / exp_name / "user" / f"user.{userId}"
    user_memory_path.write_text(responseText, encoding="utf-8")
    long_memory_path = MEMORY_ROOT / exp_name / "user-long" / f"user.{userId}"
    with open(long_memory_path, "a", encoding="utf-8") as file:
        file.write("\n=====\n")
        file.write(responseText)

def update_item_memory(pos_itemId, neg_itemId, exp_name, responseText):
    # Positive sample first (first item), negative sample second (second item)
    updated_pos_item_intro = re.split(r"The updated description of the first item is: |The updated description of the second item is: ", responseText)[1]
    updated_neg_item_intro = responseText.split("The updated description of the second item is: ")[-1]
    (MEMORY_ROOT / exp_name / "item" / f"item.{pos_itemId}").write_text(updated_pos_item_intro, encoding="utf-8")
    (MEMORY_ROOT / exp_name / "item" / f"item.{neg_itemId}").write_text(updated_neg_item_intro, encoding="utf-8")

if __name__ == "__main__":
    interDF = createInterDF(inter_data_source(mode))
    itemDF = createItemDF(item_data_source)

    initialize_memory(exp_name)
    process_interaction(interDF, itemDF, exp_name, model)
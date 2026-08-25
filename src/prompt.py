"""Prompt templates for TriRec.

This file holds two independent groups of templates:

1. TriRec main pipeline (Stage 1)
   ``item_agent_prompt`` / ``item_agent_prompt_grounded`` / ``user_agent_prompt`` /
   ``memory_integration_prompt``.  These are the templates actually used by
   ``TriRec_stage1_recall.py`` at inference time.

2. Offline AgentCF collaborative reflection
   ``user_prompt_*`` / ``item_prompt_*`` / ``system_prompt_template``.  These are
   used by ``AgentCF.py`` during the offline, label-supervised memory
   construction stage, and are independent of the main pipeline.
"""

# =============================================================================
# 1. TriRec MAIN PIPELINE (Stage 1)
# =============================================================================


def item_agent_prompt(item_title, item_description, user_description):
    """Item agent: generate a personalized promotion from item memory.

    Used by ``PROMO_MODE`` in {"full", "generic"}.  In the ``generic`` setting the
    caller passes ``GENERIC_USER_DESC`` instead of a real user profile, which
    ablates the personalization condition while keeping the promotion mechanism.
    """
    return f"""You are a product promotion agent. Your task is to create a short but appealing ad copy for the following product, highlighting its strengths and features.

Product Title: {item_title}
Product Description: {item_description}

Target User Preferences: {user_description}

Please create an ad copy of no more than 50 words, focusing on how this product meets the user's preferences. 
Avoid exaggerated marketing language, and base the ad on the actual features of the product and the specific preferences of the user.

Output format: Only output the ad copy itself, without any additional explanation or formatting."""


def item_agent_prompt_grounded(item_title, verifiable_attrs, user_description):
    """Grounded variant: only verifiable catalog attributes may be asserted.

    Used by ``PROMO_MODE == "grounded"``.  Two differences from
    ``item_agent_prompt``: (a) the factual source is the raw catalog record
    instead of the LLM-enriched item memory, so hallucinations cannot propagate
    from an earlier generation step; (b) claim types observed in the factuality
    audit are explicitly forbidden.  Personalization is preserved: the real user
    profile is still supplied and the copy is still aligned to it.
    """
    return f"""You are a product promotion agent. Write a short, appealing ad copy for the product below.

Product Title: {item_title}
Verifiable Catalog Attributes (the ONLY facts you may assert):
{verifiable_attrs}

Target User Preferences: {user_description}

STRICT GROUNDING RULES:
- You may only state facts that appear in the Verifiable Catalog Attributes above.
- Do NOT claim anything about audio quality, remastering, sound engineering, bonus tracks,
  specific track names, editions, limited/exclusive status, collectibility, curation,
  chart history, or awards. None of these are verifiable.
- Do NOT invent a physical format beyond what the Categories field states.
- You may connect the listed genre/category to the user's stated preferences.
- Subjective enthusiasm is allowed, but must not imply unverified attributes.

Write no more than 50 words, focusing on how the verifiable attributes match the user's preferences.

Output format: Only output the ad copy itself, without any additional explanation or formatting."""


def user_agent_prompt(user_description, item_ads_list, retry_hint=""):
    """User agent: score every candidate on a 0-10 scale and return strict JSON.

    Realizes the ranking operator of Eq. 5: the returned scores are sorted in
    descending order to form the Stage-1 list, and the same scores are consumed
    as r_LLM(u, i) by the Stage-2 platform utility model.

    ``retry_hint`` carries the concrete validation error (duplicated or omitted
    identifiers) back to the model when a malformed or incomplete reply is
    re-queried.
    """
    items_text = "\n".join([
        f"{i+1}. ID: {item['id']} | Title: {item['title']} | Ad: {item['ad']}"
        for i, item in enumerate(item_ads_list)
    ])

    hint_block = f"\n{retry_hint}\n" if retry_hint else ""

    return f"""You are an Amazon shopper with the following preferences and dislikes:

{user_description}

Here are several candidate items (each includes ID, Title, and Ad Copy):

{items_text}
{hint_block}
Your task:
1. Rank **all** of them according to your preferences — **use every ID exactly once** (no omission, no repetition).
2. Rank them from most to least preferred.
3. Give each item a **Relevance Score** between **0 and 10**, using a **nonlinear, human-like scale**:
   - 9–10: Perfectly fits your preferences, you would almost certainly buy it.
   - 0: Completely irrelevant or opposite to your taste.
4. The score distribution should be **nonlinear**:
   - Only a few items should get scores above 8.
   - A few should be clearly low (0–2).
5. Output the result in **strict JSON** format only, no extra text or explanation.

JSON output format (very important):
{{
  "scores": [
    {{"id": "ITEM_ID_1", "score": FLOAT}},
    {{"id": "ITEM_ID_2", "score": FLOAT}},
    ...
  ],
  "reason": "Brief explanation why you gave these scores"
}}

Rules:
- Use every ID exactly once.
- Use only the IDs listed above (do not invent or modify any).
- The list must contain exactly {len(item_ads_list)} IDs.
- The 'score' field must be numeric (0–10).
- Do not output markdown or commentary outside the JSON.
- The output will be automatically validated — if any duplication or missing ID is found, your response will be considered invalid."""


def memory_integration_prompt(item_title, current_memory, promotion_entries):
    """Item memory update of Eq. 4: fold the served promotions back into memory.

    Consumes only the audience representation each promotion was conditioned on
    and the generated text itself, so the memory records which kinds of users the
    item has addressed and how, letting later promotions build on earlier
    phrasing instead of restating the metadata.  Neither the ground-truth
    interaction nor the realized ranking is available here, so no test signal can
    enter the memory.

    Distinct from AgentCF collaborative reflection: that stage is offline,
    label-supervised and updates both the user and the item side; this mechanism
    is online, label-free and item-side only.

    Constraints:
    - Audiences must be referred to collectively, consistent with note 3 of
      ``item_prompt_template``.
    - Output stays within 50 words to match the existing item memory format.
    """
    promotions_text = "\n".join([
        f"- Audience: {e['audience']} | Angle used: {e['promotion']}"
        for e in promotion_entries
    ])

    return f"""You are maintaining the promotional profile of a product, based on the audiences it has recently addressed and the angles it used with them.

Product Title: {item_title}

Current profile:
{current_memory}

Recently served promotions:
{promotions_text}

Your task: rewrite the profile so that it records which kinds of audiences this product has addressed and which angles it used, so that future ad copies can build on that phrasing instead of restating the metadata.

Rules:
- No more than 50 words, a single paragraph, no bullet points and no headings.
- Keep the factual product attributes from the current profile; do not invent new attributes.
- Refer to audiences collectively (e.g. "listeners who prefer ..."), never as a specific individual.
- Do not state or imply that any angle succeeded or failed; no such information is available.
- Output only the rewritten profile text, without explanation or quotation marks."""


# =============================================================================
# 2. OFFLINE AgentCF COLLABORATIVE REFLECTION
# Used by AgentCF.py to build user / item memory before the main pipeline runs.
# =============================================================================


def user_prompt_system_role(user_description):
    return f"You are an Amazon buyer.\n Here is your previous self-introduction, exhibiting your past preferences and dislikes:\n '{user_description}'."

def user_prompt_template(list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"Recently, you considered choosing one item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n After comparing based on your preferences, you chose '{neg_item_title}' and rejected the other. Your explanation was:\n '{system_reason}'. \n\n However, after encountering these items, you realized you prefer '{pos_item_title}' and don't like '{neg_item_title}'.\n This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n 1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from '{pos_item_title}' and dislikes from '{neg_item_title}'. \n 3. Summarize your past preferences, merging them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes:\n 1. Your output format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it under 150 words.  \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and personalized; avoid generic preferences."

def user_prompt_template_true(list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"Recently, you considered choosing one item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n After comparing based on your preferences, you selected '{pos_item_title}' and rejected the other. Your explanation was:\n '{system_reason}'. \n\n After encountering these items, you found that you really like '{pos_item_title}' and dislike '{neg_item_title}'.\n This indicates you made a correct choice, and your judgment about your preferences was accurate. \n Your task now is to update your self-introduction to reflect your preferences and dislikes from this interaction. \n Please follow these steps: \n 1. Analyze your judgment about your preferences and dislikes from your explanation.\n 2. Identify new preferences based on '{pos_item_title}' and dislikes based on '{neg_item_title}'. \n 3. Summarize your past preferences and dislikes from your previous self-introduction, combining them with new insights while removing conflicting parts.\n 4. Update your self-introduction, starting with your new preferences, then summarizing past ones, followed by your dislikes. \n\n Important notes:\n 1. Your output format should be: 'My updated self-introduction: [Your updated self-introduction here].' \n 2. Keep it under 150 words. \n 3. Be concise and clear. \n 4. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 5. Your self-introduction should be specific and personalized; avoid generic preferences."

def item_prompt_template(user_description, list_of_item_description, pos_item_title, neg_item_title, system_reason):
    return f"User self-introduction, showing preferences and dislikes: '{user_description}'.\n Recently, the user browsed a shopping site and considered two items:\n {list_of_item_description}.\n\n He chose '{neg_item_title}' and rejected the other, explaining: '{system_reason}'. \n\n However, he prefers '{pos_item_title}' instead, indicating an unsuitable choice due to misleading descriptions. He likes '{pos_item_title}' for its features and dislikes '{neg_item_title}' for undesirable traits. Your task is to update the descriptions of these items. \n Follow these steps:\n 1. Analyze features that led to the poor choice and modify them. \n 2. Examine user preferences and dislikes; explore new features of the preferred item aligning with preferences and opposing dislikes, and do the same for the disliked item, highlighting differences. Your analysis should be thorough. \n 3. Incorporate new features into the previous descriptions, preserving valuable content while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear. \n 3. In your descriptions, refer to user preferences collectively, avoiding specific individual references, e.g., 'the user with ... preferences/dislikes'.\n 4. The updated description should not contradict the item's inherent characteristics, e.g., do not describe a thriller as having a predictably happy ending. \n 5. The updated description should highlight distinguishing features that differentiate this item from others."

def item_prompt_template_true(user_description, list_of_item_description, pos_item_title, neg_item_title):
    return f"User self-description, showcasing preferences and dislikes: '{user_description}'.\n Recently, the user browsed a shopping site and considered two items:\n {list_of_item_description}.\n\n The user chose '{pos_item_title}' for its features and rejected '{neg_item_title}' for undesirable traits. Your task is to update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, refer to preferences collectively, avoiding individual references. For example, say 'the user with ... preferences/dislikes'.\n 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items, e.g., do not describe a thriller as having a predictably happy ending."


def system_prompt_template(user_description, list_of_item_description):
    return f"You are an Amazon buyer. Here is your self-introduction, expressing your preferences and dislikes: '{user_description}'. \n\n Now, you are considering selecting an item from two candidates. The features of these items are:\n {list_of_item_description}.\n\n Please select the item that aligns best with your preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your preferences and how they relate to the item features.\n 3. Explain your choice, detailing the relationship between your preferences/dislikes and the item features. \n\n Important notes:\n 1. **Output Format:** 'Choice: [Title of the selected item] \\n Explanation: [Rationale behind your choice and reasons for rejecting the other item]'. \n 2. Do not fabricate your preferences! If your self-introduction lacks relevant details, use common knowledge to guide your decision, such as item popularity. \n 3. Select one candidate, not both. \n 4. Your explanation should be specific; general preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n 5. Base your explanation on facts. If your self-introduction doesn't specify preferences, you cannot claim your decision was influenced by them."

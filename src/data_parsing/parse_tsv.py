import json
from collections import defaultdict
import pandas as pd
import os

def parse_behavior(file_path):
    user_history=defaultdict(list)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    m=[]
    with open(file_path,'r',encoding="utf-8")as file:
        for line in file:
            parts=line.strip().split('\t')
            if len(parts) < 5:
                continue
            user_id=parts[1]
            history = parts[3].strip()
            impressions = parts[4].strip()

            if history:
                user_history[user_id].extend(history.split())

            for pair in impressions.split():
                news_id,label = pair.rsplit('-',1)
                if label == '1':
                    user_history[user_id].append(news_id)

    for user in user_history:
        seen = set()
        user_history[user] = [x for x in user_history[user] if not (x in seen or seen.add(x))]
    return user_history

def parse_news_tsv(input_path):
    records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue

            news_id = parts[0]
            category = parts[1]
            subcategory = parts[2]
            title = parts[3]
            abstract = parts[4]
            entities_json = parts[6]

            try:
                entities = json.loads(entities_json)
            except json.JSONDecodeError:
                entities = []

            records.append({
                "news_id": news_id,
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
                "entities": entities
            })

    return pd.DataFrame(records)
    
    

if __name__ == "__main__":
    # train:
    # behavior file
    input_file1 = "data/raw/MINDsmall_dev/behaviors.tsv"
    output_file1 = "data/processed/user_history.json"
    user_click_history = parse_behavior(input_file1)
    if not os.path.exists(output_file1):
        raise FileNotFoundError(f"File not found: {output_file1}")
    with open(output_file1, "w", encoding="utf-8") as f:
        json.dump(user_click_history, f, indent=2)
    
    print(f"Parsed {len(user_click_history)} users.")

    # news file
    input_file2="data/raw/MINDsmall_dev/news.tsv"
    output_file2="data/processed/news.csv"

    df = parse_news_tsv(input_file2)
    df.to_csv(output_file2, index=False)

    print(f"Parsed {len(df)} news articles.")
    print(df.head(3))    

    # test:
    # behavior file

    test_input_file1="data/raw/MINDsmall_train/behaviors.tsv"
    test_output_file1="data/processed/test/test_user_history.json"

    test_user_click_history = parse_behavior(test_input_file1)
    
    if not os.path.exists(test_output_file1):
        raise FileNotFoundError(f"File not found: {test_output_file1}")
    with open(test_output_file1, "w", encoding="utf-8") as f:
        json.dump(test_user_click_history, f, indent=2)
    
    print(f"Parsed {len(test_user_click_history)} users.")

    # news file

    test_input_file2="data/raw/MINDsmall_train/news.tsv"
    test_output_file2="data/processed/test/test_news.csv"
    df = parse_news_tsv(test_input_file2)
    df.to_csv(test_output_file2, index=False)

    print(f"Parsed {len(df)} news articles.")
    print(df.head(3))

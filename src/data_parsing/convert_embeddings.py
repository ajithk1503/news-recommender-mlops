import pickle

def load_entity_embeddings(input_path):
    embedding_dict = {}
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            entity_id = parts[0]
            vector = list(map(float, parts[1:]))
            embedding_dict[entity_id] = vector
    return embedding_dict

if __name__ == "__main__":
    # train entity
    input_file = "data/raw/MINDsmall_dev/entity_embedding.vec"
    output_file = "data/processed/entity_embedding.pkl"

    entity_embeddings = load_entity_embeddings(input_file)

    with open(output_file, 'wb') as f:
        pickle.dump(entity_embeddings, f)

    print(f"Saved {len(entity_embeddings)} entity embeddings to {output_file}")

    # test entity
    test_input_file = "data/raw/MINDsmall_train/entity_embedding.vec"
    test_output_file = "data/processed/test/test_entity_embedding.pkl"

    test_entity_embeddings = load_entity_embeddings(test_input_file)

    with open(test_output_file, 'wb') as f:
        pickle.dump(test_entity_embeddings, f)

    print(f"Saved {len(test_entity_embeddings)} entity embeddings to {test_output_file}")

    # train relation
    input_file = "data/raw/MINDsmall_dev/relation_embedding.vec"
    output_file = "data/processed/relation_embedding.pkl"

    relation_embeddings = load_entity_embeddings(input_file)

    with open(output_file, 'wb') as f:
        pickle.dump(relation_embeddings, f)

    print(f"Saved {len(relation_embeddings)} relation embeddings to {output_file}")

    # test relation
    test_input_file = "data/raw/MINDsmall_train/relation_embedding.vec"
    test_output_file = "data/processed/test/test_relation_embedding.pkl"

    test_relation_embeddings = load_entity_embeddings(test_input_file)

    with open(test_output_file, 'wb') as f:
        pickle.dump(test_relation_embeddings, f)

    print(f"Saved {len(test_relation_embeddings)} relation embeddings to {test_output_file}")



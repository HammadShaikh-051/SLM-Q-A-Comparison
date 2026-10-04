import os
from datasets import load_dataset, DatasetDict

def format_example(example):
    """
    This function processes a single dataset example.
    It extracts the first answer and constructs the final text prompt format.
    """
    # 7. For SQuAD's `answers` field, use the first answer text.
    # The 'answers' field in SQuAD is a dictionary containing lists for 'text' and 'answer_start'.
    answer = example['answers']['text'][0]
    
    # 6. Convert every example into the required prompt format
    text = f"### Context:\n{example['context']}\n\n### Question:\n{example['question']}\n\n### Answer:\n{answer}"
    
    # We return the fields we want to keep. The 'map' function will add these to our dataset.
    return {
        'context': example['context'],
        'question': example['question'],
        'answer': answer,
        'text': text
    }

def main():
    print("Loading SQuAD dataset...")
    # 1. Load the SQuAD dataset using the Hugging Face datasets library
    dataset = load_dataset("rajpurkar/squad")
    
    # 5. Use a fixed random seed of 42 so the split is reproducible
    SEED = 42
    import random
    import collections
    
    print("Grouping by context to avoid data leakage and selecting approx 10,000 examples...")
    
    # Group indices by context to prevent the same context from appearing in multiple splits
    context_to_indices = collections.defaultdict(list)
    for i, ctx in enumerate(dataset['train']['context']):
        context_to_indices[ctx].append(i)
        
    unique_contexts = list(context_to_indices.keys())
    random.seed(SEED)
    random.shuffle(unique_contexts)
    
    # 1. Select enough contexts to reach exactly or slightly over 10,000 examples
    selected_contexts = []
    total_selected_examples = 0
    for ctx in unique_contexts:
        selected_contexts.append(ctx)
        total_selected_examples += len(context_to_indices[ctx])
        if total_selected_examples >= 10000:
            break
            
    # 2. Sort selected contexts by number of examples (descending) for optimal packing
    selected_contexts.sort(key=lambda c: len(context_to_indices[c]), reverse=True)
    
    # 3. Define the exact numeric targets based on the selected total
    target_train = total_selected_examples * 0.8
    target_val = total_selected_examples * 0.1
    target_test = total_selected_examples * 0.1
    
    train_contexts = []
    val_contexts = []
    test_contexts = []
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 4. Allocate each context to the split that is currently most "starved" (largest deficit)
    for ctx in selected_contexts:
        indices = context_to_indices[ctx]
        
        # Calculate how far each split currently is from its ideal target size
        deficit_train = target_train - len(train_indices)
        deficit_val = target_val - len(val_indices)
        deficit_test = target_test - len(test_indices)
        
        # Assign the context to the split needing examples the most
        if deficit_train >= deficit_val and deficit_train >= deficit_test:
            train_contexts.append(ctx)
            train_indices.extend(indices)
        elif deficit_val >= deficit_train and deficit_val >= deficit_test:
            val_contexts.append(ctx)
            val_indices.extend(indices)
        else:
            test_contexts.append(ctx)
            test_indices.extend(indices)
            
    # 2. Print statistics
    total_train = len(train_indices)
    total_val = len(val_indices)
    total_test = len(test_indices)
    total_examples = total_train + total_val + total_test
    
    print(f"Train: {len(train_contexts)} contexts, {total_train} examples")
    print(f"Validation: {len(val_contexts)} contexts, {total_val} examples")
    print(f"Test: {len(test_contexts)} contexts, {total_test} examples")
    print(f"Total: {total_examples} examples")

    # 3. Add verification steps
    print("Verifying no context leakage across splits...")
    train_set = set(train_contexts)
    val_set = set(val_contexts)
    test_set = set(test_contexts)
    
    if not train_set.isdisjoint(val_set):
        raise ValueError("Leakage detected between train and validation splits!")
    if not train_set.isdisjoint(test_set):
        raise ValueError("Leakage detected between train and test splits!")
    if not val_set.isdisjoint(test_set):
        raise ValueError("Leakage detected between validation and test splits!")
        
    if total_examples != (total_train + total_val + total_test):
        raise ValueError("Total examples mismatch!")
        
    print("Verification passed! No context leakage detected.")
            
    # Combine the splits into a new DatasetDict structure
    final_splits = DatasetDict({
        'train': dataset['train'].select(train_indices),
        'validation': dataset['train'].select(val_indices),
        'test': dataset['train'].select(test_indices)
    })
    
    print("Formatting examples...")
    # Apply our formatting function to every example across all splits
    processed_dataset = final_splits.map(format_example)
    
    # 8. Keep these specific fields in the processed dataset
    columns_to_keep = ['context', 'question', 'answer', 'text']
    processed_dataset = processed_dataset.select_columns(columns_to_keep)
    
    # 9. Save the processed datasets so they can be loaded later by the training code
    print("Saving processed datasets...")
    # Determine the path to the 'data' directory relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, 'data', 'processed_squad')
    
    os.makedirs(output_dir, exist_ok=True)
    processed_dataset.save_to_disk(output_dir)
    
    print(f"\nData preparation complete! Saved to: {output_dir}")
    print(f"Train size: {len(processed_dataset['train'])} examples")
    print(f"Validation size: {len(processed_dataset['validation'])} examples")
    print(f"Test size: {len(processed_dataset['test'])} examples")

if __name__ == "__main__":
    main()

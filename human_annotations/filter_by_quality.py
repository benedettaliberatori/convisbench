import json 
import nltk
from nltk.corpus import wordnet
from langdetect import detect
import os
import re
from tqdm import tqdm
from collections import defaultdict
import argparse


nltk.download('wordnet')

def is_english_phrase(phrase):
    """
    Check if a phrase contains only valid English words.
    """
    words = phrase.split()
    
    return all(bool(wordnet.synsets(word.lower())) for word in words)

def looks_like_garbage(text):
    """
    Check if the text looks like garbage with regex.
    """
    return bool(re.match(r'^[^a-zA-Z0-9\s]+$', text)) or len(text) < 3 or len(set(text)) == 1


def clean_text(text):
    """
    Normalize the text removing '×' and converting to lowercase.
    """
    return text.replace('×', '').strip().lower()

def review_entries(entries, label):
    """
    Command line tool to review entries.
    Inspection on text detected as non-English.
    Three options:
    [k] keep / [e]dit / [d]elete
    """
    cleaned = []
    for i, text in enumerate(entries):
        text = clean_text(text)
        if is_english_phrase(text) or detect(text) == 'en':
            cleaned.append(text)
            continue
        
        print(f"Suspicious {label} entry [{i+1}/{len(entries)}]: '{text}'")
        action = input("Choose action: [k]eep / [e]dit / [d]elete → ").strip().lower()
        if action == 'k':
            cleaned.append(text)
        elif action == 'e':
            new_text = input("Enter corrected text: ").strip()
            if new_text:
                cleaned.append(new_text.lower())
        elif action == 'd':
            print("Deleted.")
    return cleaned


def initial_filtering(args, user_data):

    total_number_of_annotations = len(user_data)
    non_empty_custom_inputs = sum(1 for d in user_data if d.get('custom_inputs'))

    if args.verbose:
        print("Total number of annotations:", total_number_of_annotations)
        print("Number of annotations with non-empty 'custom_inputs':", non_empty_custom_inputs)
        print("Percentage of annotations with non-empty 'custom_inputs':", round((non_empty_custom_inputs / total_number_of_annotations) * 100, 2), "%")

    non_english_ids = []
    only_numbers_ids = []  
    all_commonalities = []
    all_differences = []

    for d in user_data:
        commonalities = d.get('commonalities', [])
        differences = d.get('differences', [])
        
        all_commonalities.extend(commonalities)
        all_differences.extend(differences)


        for word in commonalities:
            clean_word = word.replace('×', '').strip().lower()
            if not is_english_phrase(clean_word) and detect(clean_word) != 'en':
                non_english_ids.append(d.get('user_id')) 

            if clean_word.isdigit():
                only_numbers_ids.append(d.get('user_id'))

        for word in differences:
            clean_word = word.replace('×', '').strip().lower()
            if not is_english_phrase(clean_word) and detect(clean_word) != 'en':
                non_english_ids.append(d.get('user_id')) 

            if clean_word.isdigit():
                only_numbers_ids.append(d.get('user_id'))

    all_commonalities = [text.replace('×', '').strip().lower() for text in all_commonalities]
    all_differences = [text.replace('×', '').strip().lower() for text in all_differences]

    if args.verbose:
        print(len(non_english_ids), "annotations contain non-English words in commonalities or differences.")
        print("IDs with non-English words:", set(non_english_ids))  

    non_english_commonalities = [word for word in all_commonalities if not is_english_phrase(word)]
    non_english_differences = [word for word in all_differences if not is_english_phrase(word)]

    if args.verbose:
        print("Total commonalities:", len(all_commonalities))
        print("Total differences:", len(all_differences))
        print(len(non_english_commonalities), "non-English words in commonalities")
        print(len(non_english_differences), "non-English words in differences")
        print(len(only_numbers_ids), "annotations contain only numbers in commonalities or differences.")
        print("IDs with only number inputs:", set(only_numbers_ids))  

    filtered_data = [d for d in user_data if d.get('user_id') not in only_numbers_ids]

    return filtered_data


def manual_grammar_check(args, filtered_data):

    total_number_of_annotations = len(filtered_data)
    if args.verbose:
        print("Total number of annotations:", total_number_of_annotations)

    for d in filtered_data:
        commonalities = d.get('commonalities', [])
        differences = d.get('differences', [])
        for tag in commonalities:
            if isinstance(tag, str) and ',' in tag:
                if args.verbose:
                    print(commonalities)
                commonalities.remove(tag)
                commonalities.extend(tag.split(','))

            if looks_like_garbage(tag):
                commonalities.remove(tag)

        for tag in differences:
            if isinstance(tag, str) and ',' in tag:
                differences.remove(tag)
                differences.extend(tag.split(','))
            
            if looks_like_garbage(tag):
                differences.remove(tag)


        d['commonalities'] = commonalities
        d['differences'] = differences


    for d in tqdm(filtered_data):
        d['commonalities'] = review_entries(d.get('commonalities', []), "commonality")
        d['differences'] = review_entries(d.get('differences', []), "difference")

    return filtered_data
    

def main(args):

    with open('human_annotations/full_collection.user_inputs.json', 'r') as file:
        full_data = json.load(file)

    total_number_of_annotations = len(full_data)
    print("Total number of annotations:", total_number_of_annotations)
    print(f"\033[92mPerforming automatic filtering\033[0m")
    filtered_data = initial_filtering(args, full_data) 
    print(f"\033[92mRemaining {len(filtered_data)}/{len(full_data)} after initial automatic filtering\033[0m")
    print(f"\033[92mPerfomring manual filtering\033[0m")

    filtered_data = manual_grammar_check(args, filtered_data) # check for grammar and spelling errors
    
    count = 0
    equal_scores_by_user = defaultdict(list)

    for d in filtered_data:
        user_inputs = d.get('user_inputs', {})
        if user_inputs:
            scores = list(user_inputs.values())
            if len(set(scores)) == 1:
                user_id = d['user_id']
                equal_scores_by_user[user_id].append(d)

    total_equal = sum(len(v) for v in equal_scores_by_user.values())
    print(f"\nTotal number of annotations with all scores equal: {total_equal} out of {total_number_of_annotations}")

    total_annotations_by_user = defaultdict(int)
    for d in filtered_data:
        user_id = d['user_id']
        total_annotations_by_user[user_id] += 1

    for user_id, entries in equal_scores_by_user.items():
        if args.verbose:
            print(f"user_id {user_id} → {len(entries)} annotation(s) with all equal scores, total {total_annotations_by_user[user_id]} annotations")

    for d in filtered_data:
        commonalities = d.get('commonalities', [])
        differences = d.get('differences', [])
        if commonalities == [] and differences == []:
            if args.verbose:
                print(f"Empty commonalities and differences for user_id {d['user_id']}")
            filtered_data.remove(d)

    filtered_data = [d for d in filtered_data if d.get('user_id') != 'f923598f-12af-4b2e-8379-b9d3d2d54313']
    with open('human_annotations/filtered.user_inputs.json', 'w') as f:
        json.dump(filtered_data, f, indent=4)


    if args.verbose:
        non_empty_custom_inputs = sum(1 for d in filtered_data if d.get('custom_inputs'))
        print("Number of annotations with non-empty 'custom_inputs':", non_empty_custom_inputs)
        non_empty_custom_inputs_percentage = (non_empty_custom_inputs / total_number_of_annotations) * 100
        print("Percentage of annotations with non-empty 'custom_inputs':", round(non_empty_custom_inputs_percentage, 2), "%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during processing.",
    )
    args = parser.parse_args()
    main(args)
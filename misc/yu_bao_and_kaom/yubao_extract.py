import os
import re
import json
import multiprocessing as mp

from tqdm import tqdm


def normalize_english_meaning_punctuation(meaning):
    meaning = meaning.replace(" /", "/")
    meaning = meaning.replace("/ ", "/")
    meaning = meaning.replace(" (", "/")
    meaning = meaning.replace(") ", "/")
    return meaning


def process_json(json_path):
    with open(json_path, mode="rt", encoding="utf-8") as f:
        resp = json.load(f)
    data = resp["data"]

    mapLocation = data["mapLocation"]
    point = mapLocation["point"]
    longitude = point["longitude"]
    latitude = point["latitude"]
    location = mapLocation["location"]
    province = location["province"]
    city = location["city"]
    district = location["country"]

    meta_info = {
        "longitude": longitude,
        "latitude": latitude,
        "province": province,
        "province": province,
        "city": city,
        "district": district,
    }

    resourceList = data["resourceList"]
    assert len(resourceList) == 4

    char_info = {}

    char = resourceList[0]
    assert char["sounder"] == "老男"
    assert char["type"] == "单字"
    assert char["count"] >= 1000

    for item in char["items"]:
        zi = item["name"]
        meaning = item["en_name"]
        meaning = normalize_english_meaning_punctuation(meaning)
        assert not "#" in zi and not "#" in meaning
        key = f"#{zi}#{meaning}"
        char_info[key] = []
        records = item["records"]
        for record in records:
            sheng = record["initial"]
            yun = record["finals"]
            diao = record["tone"]
            annotation = record["memo"]
            char_info[key].append((sheng, yun, diao, annotation))

    char_alt_info = {}

    char_alt = resourceList[1]
    assert char_alt["sounder"] == "青男"
    assert char_alt["type"] == "单字"
    assert char_alt["count"] >= 1000

    for item in char_alt["items"]:
        zi = item["name"]
        meaning = item["en_name"]
        meaning = normalize_english_meaning_punctuation(meaning)
        assert not "#" in zi and not "#" in meaning
        key = f"#{zi}#{meaning}"
        char_alt_info[key] = []
        records = item["records"]
        for record in records:
            sheng = record["initial"]
            yun = record["finals"]
            diao = record["tone"]
            annotation = record["memo"]
            char_alt_info[key].append((sheng, yun, diao, annotation))

    word_info = {}

    word = resourceList[2]
    assert word["sounder"] == "方言老男"
    assert word["type"] == "词汇"
    assert word["count"] >= 1200

    for item in word["items"]:
        ci = item["name"]
        meaning = item["en_name"]
        meaning = normalize_english_meaning_punctuation(meaning)
        assert not "#" in ci and not "#" in meaning
        key = f"#{ci}#{meaning}"
        word_info[key] = []
        records = item["records"]
        for record in records:
            phonetic = record["phonetic"]
            ci = record["wording"]
            yi = record["word"]
            annotation = record["memo"]
            word_info[key].append((phonetic, ci, yi, annotation))

    syntax_info = {}

    syntax = resourceList[3]
    assert syntax["sounder"] == "方言老男"
    assert syntax["type"] == "语法"
    assert syntax["count"] >= 50

    for item in syntax["items"]:
        ju = item["name"]
        meaning = item["en_name"]
        meaning = normalize_english_meaning_punctuation(meaning)
        assert not "#" in ju and not "#" in meaning
        key = f"#{ju}#{meaning}"
        syntax_info[key] = []
        records = item["records"]
        for record in records:
            phonetic = record["phonetic"]
            ju = record["sentence"]
            annotation = record["memo"]
            syntax_info[key].append((phonetic, ju, annotation))

    return {
        "meta_info": meta_info,
        "char_info": char_info,
        "char_alt_info": char_alt_info,
        "word_info": word_info,
        "syntax_info": syntax_info,
    }


if __name__ == "__main__":

    task_list = []
    json_dir = "./yubao_raw/data"
    json_files = os.listdir(json_dir)
    for json_file in json_files:
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            task_list.append(json_path)

    all_spots = []
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap_unordered(process_json, task_list)
        for spot in tqdm(results, total=len(task_list), ncols=80):
            all_spots.append(spot)

    with open("yubao_clean/all_spots.0.json", mode="wt", encoding="utf-8") as f:
        json.dump(all_spots, f, ensure_ascii=False, indent=2)

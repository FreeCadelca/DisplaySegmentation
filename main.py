import pprint
import json


def limit_items(data, max_items=10):
    if isinstance(data, list):
        return [limit_items(item, max_items) for item in data[:max_items]]
    elif isinstance(data, dict):
        return {key: limit_items(value, max_items) for key, value in data.items()}
    else:
        return data

def read_and_print_json(file_path, max_items=10):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            limited_data = limit_items(data, max_items)
            pprint.pprint(limited_data, sort_dicts=False)
    except FileNotFoundError:
        print("Файл не найден.")
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON.")


if __name__ == "__main__":
    file_path = "Screen-segmentation-2/train/_annotations.coco.json"  # Укажите путь к вашему JSON-файлу
    read_and_print_json(file_path, max_items=20)

import logging

def load_save(saved_file):
    first_level = None
    second_level = None
    with open(saved_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        key_value = line.strip().split(":")
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            if key == "First Level":
                first_level = value
            elif key == "Second Level":
                second_level = value
    return first_level, second_level
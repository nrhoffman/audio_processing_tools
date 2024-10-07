import logging
from tools.find_first_dir import find_first_dir

def reset(original_audio):
    first_level = "/" + find_first_dir(original_audio)
    second_level = "/" + find_first_dir(original_audio + first_level)
    logging.debug(f"Rewriting save file to First Level: {first_level}, Second Level: {second_level}")
    with open("./scripts/save/create_clips.txt", "w") as file:
        file.write(f"First Level: {first_level}\n")
        file.write(f"Second Level: {second_level}\n")
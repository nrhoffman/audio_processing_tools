import logging
import os

def subtract_directories(remaining_dirs, dirs_to_remove):
    return [dir for dir in remaining_dirs if dir not in dirs_to_remove]

def find_first_dir(path, save_point = None, first_level_dirs = None):
    try:
        #Function being ran for second level dirs
        if first_level_dirs is not None:
            original_path = path
            path = os.path.join(path, first_level_dirs[0])

        entries = os.listdir(path)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(path, entry)) and entry.isdigit()]

        directories.sort(key=int)

        #No save point provided, give first directory
        if save_point is None:
            if directories:
                first_directory = directories[0]
                return first_directory
            
        #Save point provided
        else:
            remaining_directories = []
            for dir in directories:
                if ("/"+dir) == save_point:
                    break
                else:
                    remaining_directories.append(dir)
            subtracted = subtract_directories(directories, remaining_directories)

            #Returning remaining directories to process in first level
            if first_level_dirs is None:
                return subtracted
            
            #Pairing second level directories with first level directories
            else:
                dictionary = {first_level_dirs[0]: subtracted}
                for first_dir in first_level_dirs[1:]:
                    second_level_path = os.path.join(original_path, first_dir)
                    entries = os.listdir(second_level_path)
                    second_level_dirs = [entry for entry in entries if os.path.isdir(os.path.join(second_level_path, entry)) and entry.isdigit()]
                    second_level_dirs.sort(key=int)
                    dictionary[first_dir] = second_level_dirs
                return dictionary
                
    except Exception as e:
        logging.error(f"Error accessing {path}: {e}")
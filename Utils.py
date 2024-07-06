
import os
import shutil


class Util_class():
    """
    dict_in : dictionary
    list_key : list of key
    return True if ALL key in list_key is in dict
    """

    def check_key_in_dict(dict_in, list_key):
        is_in = True
        list_not_in = []
        for key in list_key:
          if key not in dict_in:
            is_in = False
            list_not_in.append(key)
        return [is_in, list_not_in]

    def same_key_in_dict(dict_in, list_key):
        is_in = True
        key_not_dict = []
        key_not_list = []

        for key in list_key:
          if key not in dict_in:
            is_in = False
            key_not_dict.append(key)

        for key in dict_in:
          if key not in list_key:
            is_in = False
            key_not_list.append(key)
        return [is_in, key_not_dict, key_not_list]

    def folder_manage(path, uniquify=True,clean=False, force=False):
        last_folder = os.path.basename(os.path.normpath(path))
        head_path = os.path.dirname(os.path.normpath(path))

        # head of path exist
        if os.path.exists(head_path):
            # path last folder not exist
            if not os.path.exists(path):
                os.makedirs(path)
                return os.path.normpath(path)
            # path last folder  exist
            else:
                if uniquify:
                    counter = 1
                    while os.path.exists(path):
                        path = head_path + "/" + last_folder  + "(" + str(counter) + ")"
                        counter += 1
                    os.makedirs(path)
                # empty last folder
                elif clean:
                    if force:
                        shutil.rmtree(path)
                        os.makedirs(path)
                    else:
                        print(f'Enter YES or Y to delete all file or directory from: {path}')
                        input_clean = input()
                        if input_clean in ["YES","Y","yes","y"]:
                            shutil.rmtree(path)
                            os.makedirs(path)
                        else:
                            raise Util_class_folder_manage_forceDelete(path)
            return os.path.normpath(path)
        else:
            print(f'Enter YES or Y to create directories: {path}')
            input_clean = input()
            if input_clean in ["YES","Y","yes","y"]:
                os.makedirs(path)
                return os.path.normpath(path)
            else:
                raise Util_class_folder_manage_dirnameNotExist(head_path)


class Util_class_folder_manage_dirnameNotExist(Exception):
    """Exception raised for errors in activation function type"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Directory name '{self.value}' not exist."


class Util_class_folder_manage_forceDelete(Exception):
    """Exception raised for errors in activation function type"""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f"Not possible force clean the folder: '{self.value}'."

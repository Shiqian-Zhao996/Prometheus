from tqdm import tqdm


def load_list(file_path):
    list_all = []
    with open(file_path, 'r') as f:
        for i in tqdm(f.readlines()):
            list_all.append(i.strip())
    
    return list_all

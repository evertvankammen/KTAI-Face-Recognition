import os
from collections import Counter


def get_image_files():
    """
            Retrieve image files from the dataset directory.

            Returns:
                list: List of file paths for images.
    """
    image_files = []
    for root, dirs, files in os.walk("./"):
        for filename in files:
            if filename.lower().endswith(('.jpg', 'jpeg', 'png')):
                image_files.append(os.path.join(root, filename))
    return image_files


if __name__ == '__main__':
    files = get_image_files()
    found_names_list_with_frame_number = []
    for file in files:
        t = file.split('./')[1]
        ts = t.split('\\')
        name = ts[0]
        number = ts[1].split('_')[1]
        print(name, number)
        found_names_list_with_frame_number.append((name, number))

    sampled = 2293
    frames = 9308
    counted = Counter(
        {'Rachel': 289, 'Chandler': 331, 'Monica': 639, 'Joey': 292, 'Phoebe': 131, 'Ross': 456, 'Unknown': 689})

    with open(f'exp_results_manual.txt', 'w') as fp:
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write('frames: ' + str(frames) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))

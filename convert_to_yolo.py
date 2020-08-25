import argparse
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


def convert_one_image(image_info, annotation_info):
    img_width = image_info['width']
    img_height = image_info['height']
    file_name = image_info['file_name']
    object_lines = []
    for anno_item in annotation_info:
        category_id = anno_item['category_id']
        xmin, ymin, w, h = anno_item['bbox']
        xmax = w + xmin
        ymax = h + ymin
        # (xmin + xmax) / 2
        x_center = (xmin + xmax) / 2.0
        # (ymin + ymax) / 2
        y_center = (ymin + ymax) / 2.0
        # normalize to 0-1
        x_center = x_center / img_width
        y_center = y_center / img_height
        w = w / img_width
        h = h / img_height

        w = 0.99999 if w > 1 else w
        h = 0.99999 if h > 1 else h
        x_center = 0.99999 if x_center > 1 else x_center
        y_center = 0.99999 if y_center > 1 else y_center

        line = f"{category_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        object_lines.append(line)
    return file_name, object_lines


def main(args):
    json_file = args.json_file  # 'visual_genome_val.json'
    image_path = args.image_path  # 'vg-images'
    out_path = args.out_path  # 'vg-val'
    print('loading data...')
    with open(json_file) as f:
        d = json.load(f)

    # names = list(range(1600))
    # for c in d['categories']:
    #     names[c['id']] = c['name']
    #
    # with open('vg_names.txt', 'w') as f:
    #     f.writelines('\n'.join(names))
    print('analyzing images...')
    images = {i['id']: i for i in d['images']}

    print('analyzing annotation...')
    annotations = defaultdict(list)
    for a in d['annotations']:
        annotations[a['image_id']].append(a)

    print('start converting...')
    for img_id, img_info in tqdm(images.items(), total=len(images)):
        anno_info = annotations[img_id]
        file_name, label_lines = convert_one_image(img_info, anno_info)
        img_file = Path(image_path) / file_name
        if img_file.exists():
            new_img_file = Path(out_path) / file_name
            print(new_img_file)
            new_img_file.parent.mkdir(parents=True, exist_ok=True)
            img_file.replace(new_img_file)
            label_file = new_img_file.with_suffix('.txt')
            label_file.write_text('\n'.join(label_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_file', type=str)
    parser.add_argument("-i", "--image_path", type=str)
    parser.add_argument("-o", "--out_path", type=str)
    main(parser.parse_args())

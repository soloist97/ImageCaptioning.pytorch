# coding: utf-8
"""
Create a reference json file used for evaluation with `coco-caption` repo.
Used when reference json is not provided, (e.g., flickr30k, or you have your own split of train/val/test)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import sys
import hashlib
from random import shuffle, seed


def main(params):

    imgs = json.load(open(params['input_json'], 'r'))['images']
    # tmp = []
    # for k in imgs.keys():
    #     for img in imgs[k]:
    #         img['filename'] = img['image_id']  # k+'/'+img['image_id']
    #         img['image_id'] = int(
    #             int(hashlib.sha256(img['image_id']).hexdigest(), 16) % sys.maxint)
    #         tmp.append(img)
    # imgs = tmp

    # create output json file
    out = {'info': {'description': 'Stanford Paragraph Dataset ({} split)'.format(params['split'])}, 'licenses': 'http://creativecommons.org/licenses/by/4.0/', 'type': 'captions'}
    out.update({'images': [], 'annotations': []})

    cnt = 0
    empty_cnt = 0
    for i, img in enumerate(imgs):
        if img['split'] != params['split']:
            continue
        out['images'].append(
            {'id': img.get('cocoid', img['imgid'])})
        for j, s in enumerate(img['sentences']):
            if len(s) == 0:
                continue
            s = ' '.join(s['tokens'])
            out['annotations'].append(
                {'image_id': out['images'][-1]['id'], 'caption': s, 'id': cnt})
            cnt += 1

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', help='input json file to process')
    parser.add_argument('--split', default='test', help='train/val/test')
    parser.add_argument('--output_json', default='data.json', help='output json file')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)


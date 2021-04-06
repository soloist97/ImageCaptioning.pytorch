"""Modified from https://github.com/lukemelas/image-paragraph-captioning/blob/master/scripts/prepro_text.py
"""
import os
import json
import argparse
from collections import OrderedDict

import spacy


spacy_en = spacy.load('en_core_web_lg')
print('SpaCy model loaded')


replacements_s = OrderedDict({
    '.T': '. T',
    '.th': '. th',
    '.A': '. A',
    '.a': '. a',
    't.v.': 'tv',
    '...': '. ',
    '..': '. ',
})


replacements_t = {
    u'½': u'half',
    u'—': u'-',
    u'™': u'',
    u'¢': u'cent',
    u'ç': u'c',
    u'û': u'u',
    u'é': u'e',
    u'°': u' degree',
    u'è': u'e',
    u'…': u'',
}


def preprocess_and_tokenize(para_str):

    for k, v in replacements_s.items():
        para_str = para_str.replace(k, v)

    # split to sentences
    tokens = list()
    for tok in spacy_en(para_str.strip()):
        if ' ' not in tok.text:
            token = tok.text.lower()
            for k, v in replacements_t.items():
                token = token.replace(k, v)
            tokens.append(token)

    return tokens

def main(args):

    # Stanford Paragraph Dataset
    paragraphs = json.load(open(os.path.join(args.para_dir, 'paragraphs_v1.json'), 'r'))
    splits = {
        'train': json.load(open(os.path.join(args.para_dir, 'train_split.json'), 'r')),
        'val': json.load(open(os.path.join(args.para_dir, 'val_split.json'), 'r')),
        'test': json.load(open(os.path.join(args.para_dir, 'test_split.json'), 'r'))
    }
    def get_split(iid):
        for s in splits.keys():
            if iid in splits[s]:
                return s
        raise Exception('iid not found in train/val/test')

    # Dump to coco format json
    images = list()
    unique_iids = list()

    for pid, anno in enumerate(paragraphs):

        # Log
        if pid % 1000 == 0:
            print('{}/{}'.format(pid, len(paragraphs)))

        # Extract info
        url = anno['url']  # original url
        filename = anno['url'].split('/')[-1]  # filename also is: str(item['image_id']) + '.jpg'
        iid = anno['image_id']  # visual genome image id (filename)

        assert filename == '{}.jpg'.format(iid)

        # Skip duplicate paragraph captions
        if iid in unique_iids:
            continue
        else:
            unique_iids.append(iid)

        # Extract paragraph and split into sentences
        raw_paragraph = anno['paragraph']

        # Write info in coco_json format
        image = dict()
        image['url'] = url
        image['filepath']  = ''                 # not relevant here (it is 'val2014' or 'train2014' for MS COCO)
        image['sentids']   = [pid]               # only one ground truth paragraph
        image['filename'] = filename
        image['imgid'] = iid
        image['split'] = get_split(iid)
        image['cocoid'] = iid
        image['id'] = iid

        # This is an array for compatibility, but it always holds exactly 1 item (a dict)
        # because we have 1 ground truth paragraph. This array hold the entire paragraph,
        # not split into sentences.
        sents = dict()
        sents['tokens'] = preprocess_and_tokenize(raw_paragraph)
        sents['raw']    = raw_paragraph
        sents['imgid']  = iid
        sents['sentid'] = pid
        sents['id']     = iid

        image['sentences'] = [sents]

        images.append(image)

    dataset_para = {
        'images': images,
        'dataset': 'para'
    }
    with open(args.output_json, 'w') as f:
        json.dump(dataset_para, f, indent=4)

    print('Finished tokenizing paragraphs.')
    print('There are {} duplicate captions.'.format(len(paragraphs) - len(unique_iids)))
    print('The dataset contains {} images and annotations'.format(len(dataset_para['images'])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--para_dir', default='data/para_data/paragraphs', help='paragraph annotation directory')
    parser.add_argument('--output_json', default='data/dataset_para.json', help='output coco format json')

    args = parser.parse_args()
    main(args)

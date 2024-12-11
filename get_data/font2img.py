# https://github.com/kaonashi-tyc/zi2zi

# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import sys
import glob
import numpy as np
import io, os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections

# reload(sys)
# sys.setdefaultencoding("utf-8")

KR_CHARSET = None
SRC_PATH = './fonts/source/'
TRG_PATH = './fonts/target/'
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
label_file = os.path.join(SCRIPT_PATH, './2350-hangul.txt')
OUTPUT_PATH = './output/'

def get_offset(ch, font, canvas_size):
    font_size = font.getsize(ch)
    font_offset = font.getoffset(ch)
    offset_x = canvas_size//2 #- font_size[0]//2
    offset_y = canvas_size//2 #- font_size[1]//2
    return [ offset_x, offset_y ]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255)).convert('L')
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def draw_example(ch, src_font, dst_font, canvas_size, src_offset, dst_offset):
    # check the filter example in the hashes or not
#     dst_hash = hash(dst_img.tobytes())
#     if dst_hash in filter_hashes:
#         return None
    dst_img = draw_single_char(ch, dst_font, canvas_size, dst_offset[0], dst_offset[1])
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img


def get_font_offset(charset, font, canvas_size):
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    font_offset = np.array([0, 0])
    count = 0
    for c in sample:
        font_img = draw_single_char(c, font, canvas_size, 0, 0)
#         font_hash = hash(font_img.tobytes())
#         if not font_hash in filter_hashes:
#             font_offset += get_offset(c, font, canvas_size)
#             count += 1
    font_offset = font_offset / count
    return font_offset

def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset[:]
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]

def select_sample(charset):
    # this returns 399 samples from KR charset
    # we selected 399 characters to sample as uniformly as possible
    # (the number of each ChoSeong is fixed to 21 (i.e., 21 Giyeok, 21 Nieun ...))
    # Given the designs of these 399 characters, the rest of Hangeul will be generated
    samples = []
    for i in range(399):
        samples.append(charset[28*i+(i%28)])
    np.random.shuffle(samples)
    return samples


def draw_handwriting(ch, src_font, canvas_size, src_offset, dst_folder):
    s = ch.decode('utf-8').encode('raw_unicode_escape').replace("\\u","").upper()
    dst_path = dst_folder + "/uni" + s + ".png"
    if not os.path.exists(dst_path):
        return
    dst_img = Image.open(dst_path)
    # check the filter example in the hashes or not
    src_img = draw_single_char(ch, src_font, canvas_size, src_offset[0], src_offset[1])
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img

def font2img(SRC_PATH, TRG_PATH, charset, char_size, canvas_size, x_offset, y_offset, sample_dir, \
             fixed_sample=False, all_sample=False, handwriting_dir=False):
    trg_fonts = glob.glob(os.path.join(TRG_PATH, '*.ttf'))
    src_font = glob.glob(os.path.join(SRC_PATH, '*.ttf'))[0]
    src_font = ImageFont.truetype(src_font, size=char_size)

#     dst_filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, 0, 0))
#     dst_offset = get_font_offset(charset, dst_font, canvas_size, dst_filter_hashes)
#     print("Src font offset : ", [x_offset, y_offset])
#     print("Dst font offset : ", dst_offset)

#     filter_hashes = set()
#     if filter_by_hash:
#         filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, dst_offset[0], dst_offset[1]))
#         print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    if handwriting_dir:
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
            
        # train dataset
        train_set = []
        for c in charset:
            e = draw_handwriting(c, src_font, canvas_size, [x_offset, y_offset], handwriting_dir)
            if e:
                code = c.decode('utf-8').encode('raw_unicode_escape').replace("\\u","").upper()
                e.save(os.path.join(sample_dir, "%d_%s_train.png" % (label, code)))
                train_set.append(c)
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
                       
        # validation dataset
        np.random.shuffle(charset)
        count = 0
        for c in charset:
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes=set())
            if e:
                code = c.decode('utf-8').encode('raw_unicode_escape').replace("\\u","").upper()
                e.save(os.path.join(sample_dir, "%d_%s_val.png" % (label, code)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
        return

    if fixed_sample:
        # train dataset
        train_set = select_sample(charset)
        for c in train_set:
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes)
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d_train.png" % (label, count)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
                       
        # validation dataset
        np.random.shuffle(charset)
        count = 0
        for c in charset:
            if count == sample_count:
                break
            if c in train_set:
                continue
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes=set())
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d_val.png" % (label, count)))
                count += 1
                if count % 100 == 0:
                    print("processed %d chars" % count)
        return

    if all_sample:
        for c in charset:
            e = draw_example(c, src_font, dst_font, canvas_size, [x_offset, y_offset], dst_offset, filter_hashes)
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d.png" % (label, count)))
                count += 1
                if count % 1000 == 0:
                    print("processed %d chars" % count)
        return

    count = 0
    font_label = 0
    for font in trg_fonts:
        font = ImageFont.truetype(font, size=char_size)
        character_count = 0
        for c in charset:
            dst_offset = (14, 12)
            e = draw_example(c, src_font, font, canvas_size, [x_offset, y_offset], dst_offset)
            if e:
                e.save(os.path.join(sample_dir, "%d_%04d.png" % (font_label, character_count)))
                character_count += 1
                count += 1
                if count % 1000 == 0:
                    print("processed %d chars" % count)
        font_label += 1
    print("processed %d chars, end" % count)


parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=45, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=64, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=14, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=8, help='y_offset')
parser.add_argument('--sample_dir', dest='sample_dir', help='directory to save examples')
parser.add_argument('--fixed_sample', dest='fixed_sample', type=int, default=0, help='pick fixed samples (399 training set, 500 test set). Note that this should not be used with --suffle.')
parser.add_argument('--all_sample', dest='all_sample', type=int, default=0, help='pick all possible samples (except for missing characters)')
parser.add_argument('--handwriting_dir', dest='handwriting_dir', default=0, help='pick handwriting samples (399 training set). Note that this should not be used with --suffle.')

args = parser.parse_args()

if __name__ == "__main__":
    with io.open(label_file, 'r', encoding='utf-8') as f:
        charset = f.read().splitlines()
    if args.shuffle:
        np.random.shuffle(charset)
    font2img(SRC_PATH, TRG_PATH, charset, args.char_size, args.canvas_size, args.x_offset, args.y_offset,
             OUTPUT_PATH, args.fixed_sample, args.all_sample, args.handwriting_dir)

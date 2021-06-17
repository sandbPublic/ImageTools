import math
from PIL import Image, ImageDraw
from typing import Tuple, List
import colormanip as cm
from dataclasses import dataclass

FILE_PREFIX = 'imageA'

@dataclass
class HSV:
    h: int
    s: int
    v: int

    def to_string_from_bytes(self):
        return f'h{int(45 * self.h / 32) % 360:03d}_s{int(25 * self.s / 64):03d}_v{int(25 * self.v / 64):03d}'


def create_spectra():
    im = Image.new(mode="HSV", size=(256, 256), color=(0, 0, 0))
    for x in range(im.size[0]):
        for y in range(im.size[1] // 2):
            im.putpixel((x, y), (x, 255, 255))
        for y in range(im.size[1] // 2, im.size[1]):
            im.putpixel((x, y), (x, 255, 510 - 2 * y))
    im = im.convert('RGB')
    im.save('spectrum.png')

    im = im.convert('HSV')
    im = cm.map_pixels(im, lambda c: cm.box_swap_hsv(c, [100, 156, 0, 255, 0, 255], [95, 161, -1, 256, -1, 256]),
                       [0, 255, 64, 255])
    im = im.convert('RGB')
    im.save('spectrum2.png')

    im = Image.open('spectrum.png')
    im = im.convert('HSV')
    im = cm.map_pixels(im, lambda c: cm.box_swap_hsv(c, [100, 156, 0, 255, 0, 255], [95, 161, -1, 256, -1, 256]),
                       [0, 255, 64, 255])
    im = im.convert('RGB')
    im.save('spectrum3.png')


def create_histograms(im, hsv_list=None, top_pixel=255):
    im = im.convert('HSV')

    hues = [0] * 256
    sats = [0] * 256
    vals = [0] * 256

    if hsv_list is None:
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                hsv = im.getpixel((x, y))
                hues[hsv[0]] += 1
                sats[hsv[1]] += 1
                vals[hsv[2]] += 1
    else:
        for hsv in hsv_list:
            hues[hsv[0]] += 1
            sats[hsv[1]] += 1
            vals[hsv[2]] += 1

    histo_hues = Image.new(mode='HSV', size=(256, top_pixel + 1), color=(0, 0, 255))  # white
    histo_sats = Image.new(mode='HSV', size=(256, top_pixel + 1), color=(0, 0, 0))  # black
    histo_vals = Image.new(mode='HSV', size=(256, top_pixel + 1), color=(80, 255, 255))  # green

    drawh = ImageDraw.Draw(histo_hues)
    draws = ImageDraw.Draw(histo_sats)
    drawv = ImageDraw.Draw(histo_vals)

    h_scale = top_pixel / max(hues)
    s_scale = top_pixel / max(sats)
    v_scale = top_pixel / max(vals)

    for i in range(256):
        drawh.line([i, top_pixel, i, top_pixel - int(hues[i] * h_scale)], (i, 255, 255))
        draws.line([i, top_pixel, i, top_pixel - int(sats[i] * s_scale)], (0, i, 255))
        drawv.line([i, top_pixel, i, top_pixel - int(vals[i] * v_scale)], (0, 0, i))

    histo_hues = histo_hues.convert('RGB')
    histo_hues.save(FILE_PREFIX + '_histogram_hues.png')
    histo_sats = histo_sats.convert('RGB')
    histo_sats.save(FILE_PREFIX + '_histogram_sats.png')
    histo_vals = histo_vals.convert('RGB')
    histo_vals.save(FILE_PREFIX + '_histogram_vals.png')


def create_scatterplots(im, hsv_list=None):
    im = im.convert('HSV')

    scatter_hs = Image.new(mode='HSV', size=(256, 256), color=(0, 0, 0))  # black, pixels in plot have v set to 255
    scatter_sv = Image.new(mode='HSV', size=(256, 256), color=(80, 255, 255))  # green, pixels have h set to 0
    scatter_hv = Image.new(mode='HSV', size=(256, 256), color=(0, 0, 255))  # white, pixels have s set to 255

    if hsv_list is None:
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                hsv = im.getpixel((x, y))
                scatter_hs.putpixel((hsv[0], hsv[1]), (hsv[0], hsv[1], 255))
                scatter_sv.putpixel((hsv[1], hsv[2]), (0, hsv[1], hsv[2]))
                scatter_hv.putpixel((hsv[0], hsv[2]), (hsv[0], 255, hsv[2]))
    else:
        for hsv in hsv_list:
            scatter_hs.putpixel((hsv[0], hsv[1]), (hsv[0], hsv[1], 255))
            scatter_sv.putpixel((hsv[1], hsv[2]), (0, hsv[1], hsv[2]))
            scatter_hv.putpixel((hsv[0], hsv[2]), (hsv[0], 255, hsv[2]))

    scatter_hs = scatter_hs.convert('RGB')
    scatter_hs.save(FILE_PREFIX + '_scatter_hs.png')
    scatter_sv = scatter_sv.convert('RGB')
    scatter_sv.save(FILE_PREFIX + '_scatter_sv.png')
    scatter_hv = scatter_hv.convert('RGB')
    scatter_hv.save(FILE_PREFIX + '_scatter_hv.png')


# mask white: force exclusion of this pixel
def pixel_excluded(rgb):
    return rgb[0] >= 128  # too red to be either green or black


# mask black: force inclusion of this pixel
def pixel_included(rgb):
    return rgb[1] < 128  # not enough green to be white or green


# construct a compound mask:
# included or excluded in the initial mask remains the same
# conditional in initial mask and in source box = green
# conditional in initial mask but not in source box = red
def create_composite_mask(im,
                          mask,
                          byte_source_box: List[int],
                          out_source_color=(255, 0, 0),
                          in_source_color=(0, 255, 0)):
    mask = mask.convert('RGB')
    im = im.convert('HSV')
    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if pixel_excluded(mask_rgb) or pixel_included(mask_rgb):
                mask.putpixel(coord, mask_rgb)
                continue

            image_hsv = im.getpixel(coord)
            image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])

            if byte_source_box[0] <= image_hue <= byte_source_box[1] and \
                    byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and \
                    byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]:
                mask.putpixel(coord, in_source_color)
            else:
                mask.putpixel(coord, out_source_color)

    mask.save(FILE_PREFIX + 'mask_compound.png')


# converts very unsaturated or very dark pixels only which are outside desired hue range
# which otherwise in rgb would have arbitrary or 0 hue when they shouldn't
def convert_greys(im,
                  coord_list,
                  hsv_list,
                  hue_min,
                  hue_max,
                  sat_min,
                  val_min):
    hue_min = int(hue_min * 32 / 45)  # convert from 360 to byte
    hue_max = int(hue_max * 32 / 45)  # convert from 360 to byte
    sat_min = int(sat_min * 64 / 25)  # convert from 100 to byte
    val_min = int(val_min * 64 / 25)  # convert from 100 to byte

    im = im.convert('HSV')
    marker = Image.new(mode="L", size=im.size, color=0)

    pixels_converted = 0
    for coord, hsv in zip(coord_list, hsv_list):
        marker.putpixel(coord, 127)

        if hsv[1] < sat_min or hsv[2] < val_min:
            new_hue = cm.hue_nearest_bands(hsv[0], hue_min, hue_max)

            needs_change = False
            if new_hue < hue_min:
                new_hue = hue_min
                needs_change = True

            if new_hue > hue_max:
                new_hue = hue_max
                needs_change = True

            if needs_change:
                # need some saturation and value to avoid degenerate HSV regions so hue can be reconstructed
                im.putpixel(coord, (new_hue % 256, max(2, hsv[1]), max(2, hsv[2])))
                marker.putpixel(coord, 255)
                pixels_converted += 1

    im = im.convert('RGB')
    im.save(FILE_PREFIX + f'setWhiteBlackTo{hue_min:3d}to{hue_max:3d}.png')
    marker.save(FILE_PREFIX + "_marker.png")
    print(f'{pixels_converted} pixels converted')


def create_color_permutations():
    im = Image.open(FILE_PREFIX + '.png')
    im = cm.linear_map_colors(im, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    im.save(FILE_PREFIX + '_rbg.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = cm.linear_map_colors(im, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    im.save(FILE_PREFIX + '_grb.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = cm.linear_map_colors(im, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    im.save(FILE_PREFIX + '_brg.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = cm.linear_map_colors(im, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    im.save(FILE_PREFIX + '_gbr.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = cm.linear_map_colors(im, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    im.save(FILE_PREFIX + '_bgr.png')


def create_hue_rotations(rotation_amount=24):
    for i in range(rotation_amount, 256, rotation_amount):
        im = Image.open(FILE_PREFIX + '.png')
        im = im.convert('HSV')
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                hsv = im.getpixel((x, y))
                im.putpixel((x, y), ((hsv[0] + i) % 256, hsv[1], hsv[2]))
        im = im.convert('RGB')
        im.save(FILE_PREFIX + f'_rot{i:03d}.png')


def preprocess_image_and_mask(im, mask, byte_source_box=None):
    coord_list = []
    hsv_list = []
    hue_counts = [0] * 256
    sat_sum = 0
    val_sum = 0

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if pixel_excluded(mask_rgb):
                continue

            image_hsv: Tuple[int, int, int] = im.getpixel(coord)

            # now either included or conditional 
            # if no source box provided, treat as if it is in i.e. OK
            if not pixel_included(mask_rgb) and byte_source_box is not None:
                image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])
                if not (byte_source_box[0] <= image_hue <= byte_source_box[1] and
                        byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and
                        byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]):
                    continue  # conditional mask region and outside source box (failed)

            coord_list.append(coord)
            hsv_list.append(image_hsv)

            # dark and desat don't contribute as much to hue, weight appropriately
            hue_counts[image_hsv[0]] += math.sqrt(image_hsv[1]*image_hsv[2])
            sat_sum += image_hsv[1]
            val_sum += image_hsv[2]

    included_pixel_count = len(coord_list)
    print(f'mask_list size {included_pixel_count}')

    avg_hue = 0
    hue_sum = sum(hue_counts)
    # hue is cyclical, so average depends on the cycle midpoint, but should stabilize
    hue_midpoint = 128
    while hue_midpoint != avg_hue:  # repeat until stable
        hue_midpoint = avg_hue
        for hue in range(hue_midpoint - 128, hue_midpoint + 128):
            avg_hue += hue_counts[hue % 256] * hue
        avg_hue = int(avg_hue / hue_sum)

    avg_hsv = HSV(avg_hue % 256,
                  sat_sum // included_pixel_count,
                  val_sum // included_pixel_count)
    print(f'average hsv: {avg_hsv.to_string_from_bytes()}')

    return coord_list, hsv_list, avg_hsv


num_hues = 18
# even_spaced_hues = [hue for hue in range(0, 360, 360 // num_hues)]
uneven_spaced_hues = [hue for hue in range(-60, 120, 270 // num_hues)]
uneven_spaced_hues.extend([hue for hue in range(120, 300, 540 // num_hues)])


def target_boxes(source_box, hue_range_scale=1.0):
    trgt_bxs = []

    hue_range = int((source_box[1] - source_box[0]) * hue_range_scale)

    trgt_bxs.append([source_box[0], source_box[1], 0, 10, 0, 35, 'black'])
    trgt_bxs.append([source_box[0], source_box[1], 0, 10, 30, 70, 'grey'])
    trgt_bxs.append([source_box[0], source_box[1], 0, 10, 65, 100, 'white'])

    def convert_range(source_lo, source_hi, target_mid):
        radius = (source_hi - source_lo) // 2
        target_range = [target_mid - radius, target_mid + radius]
        overflow = target_range[1] - 100
        if overflow > 0:
            target_range[1] = 100
            target_range[0] += overflow  # need to move lo point up to move midpoint since hi is maxed
        if target_range[0] < 0:
            target_range[1] += target_range[0]  # need to move hi point down to move midpoint since lo is minimal
            target_range[0] = 0
        target_range[0] = min(90, target_range[0])
        target_range[1] = max(10, target_range[1])
        return target_range

    hi_sat_range = convert_range(source_box[2], source_box[3], 70)
    lo_sat_range = convert_range(source_box[2], source_box[3], 30)

    hi_val_range = convert_range(source_box[4], source_box[5], 70)
    lo_val_range = convert_range(source_box[4], source_box[5], 30)

    for hue in uneven_spaced_hues:
        hue_start = hue - hue_range // 2
        hue_end = hue_start + hue_range

        trgt_bxs.append(
            [hue_start, hue_end, source_box[2], source_box[3], source_box[4], source_box[5], 'sameSV'])
        trgt_bxs.append(
            [hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], hi_val_range[0], hi_val_range[1], 'bright'])
        trgt_bxs.append(
            [hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], hi_val_range[0], hi_val_range[1], 'light'])
        trgt_bxs.append(
            [hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], lo_val_range[0], lo_val_range[1], 'dark'])
        trgt_bxs.append(
            [hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], lo_val_range[0], lo_val_range[1], 'dull'])

    return trgt_bxs


def create_conversion_guide(converted_hsv: List[HSV], column_size=16):
    im = Image.new(mode="HSV", size=(256, 6 * column_size), color=(0, 0, 0))
    drawer = ImageDraw.Draw(im)

    for x in range(256):
        drawer.line([x, 0 * column_size, x, 1 * column_size - 1], (x, 255, 255))
        drawer.line([x, 1 * column_size, x, 2 * column_size - 1], (converted_hsv[x].h, 255, 255))
        drawer.line([x, 2 * column_size, x, 3 * column_size - 1], (0, x, 255))
        drawer.line([x, 3 * column_size, x, 4 * column_size - 1], (0, converted_hsv[x].s, 255))
        drawer.line([x, 4 * column_size, x, 5 * column_size - 1], (0, 0, x))
        drawer.line([x, 5 * column_size, x, 6 * column_size - 1], (0, 0, converted_hsv[x].v))

    im = im.convert('RGB')
    im.save(f'conversion_guide.png')


# precompute conversions for each possible byte for HSV
def box_swap_conversion(byte_source_box, target_box):
    converted_hsv = []

    byte_target_box = cm.convert_hsv_box_to_bytes(target_box)
    for b in range(256):
        hue = cm.hue_nearest_bands(b, byte_source_box[0], byte_source_box[1])

        # scale coordinates to 0,1 relative to source box
        scale = cm.weighted_average_factor(hue, byte_source_box[0], byte_source_box[1])
        # scale to absolute coordinates using target box
        converted_hue = int(cm.linear_combination(byte_target_box[0], byte_target_box[1], scale)) % 256

        scale = cm.weighted_average_factor(b, byte_source_box[2], byte_source_box[3])
        converted_sat = int(cm.linear_combination(byte_target_box[2], byte_target_box[3], scale))

        scale = cm.weighted_average_factor(b, byte_source_box[4], byte_source_box[5])
        converted_val = int(cm.linear_combination(byte_target_box[4], byte_target_box[5], scale))

        converted_hsv.append(HSV(converted_hue, max(min(converted_sat, 255), 0), max(min(converted_val, 255), 0)))

    return converted_hsv


# hue_contraction = K: new(h) = h + t - a + aK - hK
# value of 1 will contract to a point at target_hsv[0], 0 does nothing
# S and V multiply (or multiply complement) to change average to target
def average_to_target_conversion(avg_hsv: HSV, target_hsv: HSV, hue_contraction=0.0):

    # fails only if target and avg == 0; if avg == 255, target <= avg
    def SV_mult(x: int, avg: int, target: int):
        if target <= avg:
            return x * target // avg
        else:
            # if target > avg, flip up with down, eg:
            # target = 75% and avg = 50%, then 10% -> 55% not 15%, 0% -> 50% not 0%, 100% -> 100% not 150%
            # 75% -> ~25%, 50% -> ~50%, multiplier = 0.5
            # 10% -> ~90%, * 0.5 = ~45% -> 55%
            # replace every number by its complement
            return 255 - (255 - x) * (255 - target) // (255 - avg)

    hue_delta = target_hsv.h - avg_hsv.h
    return [HSV(int((b + hue_delta + hue_contraction * (avg_hsv.h - b))) % 256,
                SV_mult(b, avg_hsv.s, target_hsv.s),
                SV_mult(b, avg_hsv.v, target_hsv.v))
            for b in range(256)]


def create_color_swap(im, converted_hsv, coord_list, hsv_list, filename):
    im = im.convert('HSV')

    for coord, hsv in zip(coord_list, hsv_list):
        im.putpixel(coord, (converted_hsv[hsv[0]].h, converted_hsv[hsv[1]].s, converted_hsv[hsv[2]].v))

    im = im.convert('RGB')
    im.save(filename)
    print(filename)


def create_edge_detection(im, buffer=1):
    im = im.convert('RGB')
    edges = Image.new(mode='RGB', size=im.size, color=(0, 0, 0))

    cached_roots = [math.sqrt(x) for x in range(buffer * buffer * 2 + 1)]
    greatest_contrast = 0  # largest edge for each of rgb should display as 255
    unscaled_values = [[[] for y in range(im.size[1])] for x in range(im.size[0])]

    print("Creating edges...")
    for x in range(buffer, im.size[0] - buffer):
        for y in range(buffer, im.size[1] - buffer):
            total_contrast = [0, 0, 0]
            current_pixel = im.getpixel((x, y))
            for i in range(-buffer, buffer + 1):
                for j in range(-buffer, buffer + 1):
                    square_dist = i * i + j * j
                    if square_dist == 0:
                        continue

                    neighbor = im.getpixel((x + i, y + j))
                    for c in range(3):
                        total_contrast[c] += abs(neighbor[c] - current_pixel[c]) / cached_roots[square_dist]

            unscaled_values[x][y] = total_contrast

            for c in total_contrast:
                if greatest_contrast < c:
                    greatest_contrast = c

    print("Scaling edges...")
    for x in range(buffer, im.size[0] - buffer):
        for y in range(buffer, im.size[1] - buffer):
            edges.putpixel((x, y), tuple([int(c * 255 / greatest_contrast) for c in unscaled_values[x][y]]))

    edges.save(FILE_PREFIX + "_edges.png")


# todo get filenames from input()
def run():
    try:
        im = Image.open(FILE_PREFIX + 'a.png')  # alternate or edited image
        print('using alternate image')
    except FileNotFoundError:
        im = Image.open(FILE_PREFIX + '.png')

    # create_histograms(im)
    # create_scatterplots(im)
    # create_edge_detection(im, 2)
    # exit(0)

    im = im.convert('HSV')

    try:
        mask = Image.open((FILE_PREFIX + 'mask2.png'))
        print('using alternate mask')
    except FileNotFoundError:
        mask = Image.open((FILE_PREFIX + 'mask.png'))
    mask = mask.convert('RGB')

    source_box = [-10, 40, 0, 100, 0, 100]
    byte_source_box = cm.convert_hsv_box_to_bytes(source_box)

    # create_composite_mask(im, mask, byte_source_box)
    # exit(0)

    coord_list, hsv_list, avg_hsv = preprocess_image_and_mask(im, mask, byte_source_box)
    # exit(0)

    # create_histograms(im, hsv_list)
    # create_scatterplots(im, hsv_list)
    # exit(0)

    # convert_greys(im, coord_list, hsv_list, source_box[0], source_box[1], 9, 9)
    # exit(0)

    SVscale = 0.75
    required_range = 32
    low_sat = min(int(avg_hsv.s * SVscale), 128 - required_range)
    hi_sat = max(int(255 - (255 - avg_hsv.s) * SVscale), 128 + required_range)
    low_val = min(int(avg_hsv.v * SVscale), 128 - required_range)
    hi_val = max(int(255 - (255 - avg_hsv.v) * SVscale), 128 + required_range)

    target_sats = [low_sat, low_sat, hi_sat, hi_sat, avg_hsv.s]
    target_vals = [low_val, hi_val, low_val, hi_val, avg_hsv.v]

    all_targets = [HSV(avg_hsv.h, 32, 48), HSV(avg_hsv.h, 16, 240)]
    for hue in uneven_spaced_hues:
        for sat, val in zip(target_sats, target_vals):
            all_targets.append(HSV(int(hue * 32/45), sat, val))

    for target in all_targets:
        create_color_swap(im.copy(),
                          average_to_target_conversion(avg_hsv, target, 0.75),
                          coord_list,
                          hsv_list,
                          FILE_PREFIX + f'_{target.to_string_from_bytes()}.png')


if __name__ == '__main__':
    run()
    # cProfile.run('run()', sort='tottime')

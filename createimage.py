from PIL import Image, ImageDraw
from typing import Tuple, List, Callable
import colormanip as cm

FILE_PREFIX = 'image13'


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


def create_histograms(im, top_pixel=255):
    im = im.convert('HSV')

    hues = [0] * 256
    sats = [0] * 256
    vals = [0] * 256

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            hsv = im.getpixel((x, y))

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


def create_scatterplots(im):
    im = im.convert('HSV')

    scatter_hs = Image.new(mode='HSV', size=(256, 256), color=(0, 0, 0))  # black, pixels in plot have v set to 255
    scatter_sv = Image.new(mode='HSV', size=(256, 256), color=(80, 255, 255))  # green, pixels have h set to 0
    scatter_hv = Image.new(mode='HSV', size=(256, 256), color=(0, 0, 255))  # white, pixels have s set to 255

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            hsv = im.getpixel((x, y))
            scatter_hs.putpixel((hsv[0], hsv[1]), (hsv[0], hsv[1], 255))
            scatter_sv.putpixel((hsv[1], hsv[2]), (0, hsv[1], hsv[2]))
            scatter_hv.putpixel((hsv[0], hsv[2]), (hsv[0], 255, hsv[2]))

    scatter_hs = scatter_hs.convert('RGB')
    scatter_hs.save(FILE_PREFIX + '_scatter_hs.png')
    scatter_sv = scatter_sv.convert('RGB')
    scatter_sv.save(FILE_PREFIX + '_scatter_sv.png')
    scatter_hv = scatter_hv.convert('RGB')
    scatter_hv.save(FILE_PREFIX + '_scatter_hv.png')


# construct a compound mask:
# included or excluded in the initial mask remains the same
# conditional in initial mask and in source box = green
# conditional in initial mask but not in source box = red
def create_composite_mask(im, mask, byte_source_box: List[int],
                          out_source_color=(255, 0, 0), in_source_color=(0, 255, 0)):
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
def convert_greys(im, bounding_box, mask_array,
                  hue_min, hue_max, sat_min, val_min):
    hue_min = int(hue_min * 32 / 45)  # convert from 360 to byte
    hue_max = int(hue_max * 32 / 45)  # convert from 360 to byte
    sat_min = int(sat_min * 64 / 25)  # convert from 100 to byte
    val_min = int(val_min * 64 / 25)  # convert from 100 to byte

    im = im.convert('HSV')
    marker = Image.new(mode="L", size=im.size, color=0)

    pixels_converted = 0
    for x in range(bounding_box[0], bounding_box[1]):
        for y in range(bounding_box[2], bounding_box[3]):
            if mask_array[x][y]:
                coord = (x, y)
                hsv = im.getpixel(coord)
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


# mask white: force exclusion of this pixel
def pixel_excluded(rgb):
    return rgb[0] >= 128  # too red to be either green or black


# mask black: force inclusion of this pixel
def pixel_included(rgb):
    return rgb[1] < 128  # not enough green to be white or green


num_hues = 18
# even_spaced_hues = [hue for hue in range(0, 360, 360 // num_hues)]
uneven_spaced_hues = [hue for hue in range(-60, 120, 270 // num_hues)]
uneven_spaced_hues.extend([hue for hue in range(120, 300, 540 // num_hues)])


def preprocess_image_and_mask(im, mask, byte_source_box=None):
    bounding_box = [mask.size[0], 0, mask.size[1], 0]
    mask_array = [[False for y in range(mask.size[1])] for x in range(mask.size[0])]
    coord_list = []
    hsv_list = []

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if pixel_excluded(mask_rgb):
                continue
            
            image_hsv = im.getpixel(coord)

            # now either included or conditional 
            # if no source box provided, treat as if it is in i.e. OK
            if not pixel_included(mask_rgb) and byte_source_box is not None:
                image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])
                if not (byte_source_box[0] <= image_hue <= byte_source_box[1] and
                        byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and
                        byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]):
                    continue  # conditional mask region and outside source box (failed)

            mask_array[x][y] = True
            coord_list.append(coord)
            hsv_list.append(image_hsv)
            if bounding_box[0] > x:
                bounding_box[0] = x
            if bounding_box[1] < x:
                bounding_box[1] = x
            if bounding_box[2] > y:
                bounding_box[2] = y
            if bounding_box[3] < y:
                bounding_box[3] = y

    print(bounding_box)
    print(f'bounding_box size {(bounding_box[1] - bounding_box[0]) * (bounding_box[3] - bounding_box[2])}')
    print(f'mask_list size {len(coord_list)}')

    # necessary padding
    bounding_box[1] += 1
    bounding_box[3] += 1

    return bounding_box, mask_array, coord_list, hsv_list


def target_boxes(source_box):
    target_boxes = []

    hue_range = int((source_box[1] - source_box[0]) / 1)

    target_boxes.append([source_box[0], source_box[1], 0, 10, 0, 35, 'black'])
    target_boxes.append([source_box[0], source_box[1], 0, 10, 30, 70, 'grey'])
    target_boxes.append([source_box[0], source_box[1], 0, 10, 65, 100, 'white'])

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

        target_boxes.append(
            [hue_start, hue_end, source_box[2], source_box[3], source_box[4], source_box[5], 'sameSV'])
        target_boxes.append(
            [hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], hi_val_range[0], hi_val_range[1], 'bright'])
        target_boxes.append(
            [hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], hi_val_range[0], hi_val_range[1], 'light'])
        target_boxes.append(
            [hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], lo_val_range[0], lo_val_range[1], 'dark'])
        target_boxes.append(
            [hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], lo_val_range[0], lo_val_range[1], 'dull'])

    return target_boxes


def create_color_swap(im, target_box, byte_source_box, coord_list, hsv_list):
    im = im.convert('HSV')

    filename = FILE_PREFIX + f'_h{target_box[0] % 360:03d}_{target_box[1] % 360:03d}' \
                             f'_s{target_box[2]:03d}_{target_box[3]:03d}' \
                             f'_v{target_box[4]:03d}_{target_box[5]:03d}' \
                             f'_{target_box[6]}.png'
    byte_target_box = cm.convert_hsv_box_to_bytes(target_box)

    # precompute conversions for each possible byte for HSV
    converted_hsv = [[], [], []]
    for b in range(256):
        for i in range(3):
            # scale coordinates to 0,1 relative to source box
            scale = cm.weighted_average_factor(b, byte_source_box[2 * i], byte_source_box[2 * i + 1])
            # scale to absolute coordinates using target box
            converted = int(
                cm.linear_combination(byte_target_box[2 * i], byte_target_box[2 * i + 1], scale)) % 256
            converted_hsv[i].append(converted)

    for coord, hsv in zip(coord_list, hsv_list):
        im.putpixel(coord, (converted_hsv[0][hsv[0]], converted_hsv[1][hsv[1]], converted_hsv[2][hsv[2]]))

    im = im.convert('RGB')
    im.save(filename)
    print(filename)


def run():
    try:
        im = Image.open(FILE_PREFIX + 'a.png')  # alternate or edited image
        print('using alternate image')
    except FileNotFoundError:
        im = Image.open(FILE_PREFIX + '.png')
    im = im.convert('HSV')

    # create_histograms(im)
    # create_scatterplots(im)
    # exit(0)

    try:
        mask = Image.open((FILE_PREFIX + 'mask2.png'))
        print('using alternate mask')
    except FileNotFoundError:
        mask = Image.open((FILE_PREFIX + 'mask.png'))
    mask = mask.convert('RGB')

    source_box = [18, 66, 10, 86, 40, 100]
    byte_source_box = cm.convert_hsv_box_to_bytes(source_box)

    # create_composite_mask(im, mask, byte_source_box)
    # exit(0)

    bounding_box, mask_array, coord_list, hsv_list = preprocess_image_and_mask(im, mask, byte_source_box)

    # convert_greys(im, bounding_box, mask_array, source_box[0], source_box[1], 9, 9)
    # exit(0)

    for target_box in target_boxes(source_box):
        try:
            im = Image.open(FILE_PREFIX + 'a.png')  # alternate or edited image
        except FileNotFoundError:
            im = Image.open(FILE_PREFIX + '.png')

        create_color_swap(im, target_box, byte_source_box, coord_list, hsv_list)


if __name__ == '__main__':
    run()
    # cProfile.run('run()', sort='tottime')

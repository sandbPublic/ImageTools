from PIL import Image, ImageDraw
from typing import Tuple, List, Callable
import cProfile


# expand range for colors so that there is a point with value of 0 and a point with value 255 for each of R,G,B.
# scale from [0, 1] dictates the extent of the effect
def extend_colors(im, scale=1):
    im = im.convert('RGB')

    min_color = list(im.getpixel((0, 0)))
    max_color = list(im.getpixel((0, 0)))
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            pixel = im.getpixel((x, y))
            for i, c in enumerate(pixel):
                if max_color[i] < c:
                    max_color[i] = c
                if min_color[i] > c:
                    min_color[i] = c

    color_range = [0, 0, 0]
    color_scale = [1, 1, 1]
    min_after_scale = [c for c in min_color]
    for i in range(3):
        print(f'extending {i} {min_color[i]} {max_color[i]}')
        color_range[i] = max_color[i] - min_color[i]
        if color_range == 0:
            continue
        color_scale[i] = 1 + (255.0/color_range[i] - 1)*scale
        min_after_scale[i] = min_color[i]*(1-scale)

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            color = list(im.getpixel((x, y)))
            for i in range(3):
                color[i] -= min_color[i]
                color[i] *= color_scale[i]
                color[i] += min_after_scale[i]
                color[i] = int(color[i])
            im.putpixel((x, y), tuple(color))

    return im


# each rgb value is mapped by linear combination to new value
def linear_map_colors(im, matrix, normalize=True):
    if im.mode != 'RGB':
        print(f'converting mode from {im.mode} to rgb')
        im = im.convert('RGB')

    # normalize mapping
    if normalize:
        for i, row in enumerate(matrix):
            s = sum(row)
            if s != 0:
                matrix[i] = [m / s for m in row]
            else:
                matrix[i] = [0, 0, 0]

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            old_color = im.getpixel((x, y))
            new_color = [old_color[0] * row[0] + old_color[1] * row[1] + old_color[2] * row[2] for row in matrix]
            im.putpixel((x, y), tuple(int(c) for c in new_color))

    return im


def map_pixels(im, mapper: Callable[[Tuple], Tuple], bounding_box: List[int]):
    for x in range(bounding_box[0], bounding_box[1]):
        for y in range(bounding_box[2], bounding_box[3]):
            im.putpixel((x, y), mapper(im.getpixel((x, y))))

    return im


# for swapping from a range to a point instead of point to point
# reverse start/finish to invert
# ranges from 0 to 1
def weighted_average_factor(x, start, finish):
    if start > finish:
        return 1 - weighted_average_factor(x, finish, start)
    
    if x >= finish:
        return 1
    if x <= start:
        return 0
    return (x - start)/(finish - start)


def linear_combination(a, b, factor):
    return a*(1-factor) + b*factor


def convert_hsv_box_to_bytes(box: List[int]):
    return [box[0] * 32/45, box[1] * 32/45,
            box[2] * 64/25, box[3] * 64/25,
            box[4] * 64/25, box[5] * 64/25]


# for any color in source box, convert to color in target box with same relative coordinates scaled to each box
# for example point  (10,20,30) in source_box [  0,20,   0,60, 0,120] is mapped to
#                   (110,60, 3) in target_box [100,120, 50,80, 4,0]
# if in box_finish but not box_start, diminish change
# boxes can be oriented backward
# can take box values from 255,255,255 (pre-convert from 360, 100, 100)
# hue modular arithmetic must be accounted for
# min should always be less than max for ranges, e.g. convert (200, 20) to (200, 20+256)
# assume true for input boxes
def box_swap_hsv(color: Tuple[int], source_box: List, target_box: List) -> Tuple[int]:
    return_color = list(color)

    # when considering a point relative to a range, abs(min - point) <= 128, abs(max - point) <= 128
    while return_color[0] < source_box[0] - 128:
        return_color[0] += 256
    while return_color[0] > source_box[1] + 128:
        return_color[0] -= 256

    for i in range(3):
        if return_color[i] < source_box[2 * i] or source_box[2 * i + 1] < return_color[i]:
            return color

    # scale coordinates to 0,1 relative to source box
    scales = tuple(weighted_average_factor(color[i], source_box[2 * i], source_box[2 * i + 1]) for i in range(3))
    # scale to absolute coordinates using target box
    return_color = [int(linear_combination(target_box[2*i], target_box[2*i+1], scales[i])) for i in range(3)]
    return_color[0] %= 256

    return tuple(return_color)


# values are considered outside if the sum of their deviations from an inner box are too large
# in 2d, forms a rectangle with vertices cut to form an octagon
# low laxness cuts vertices more:
# laxness of 0   -> never inside
# laxness of 1   -> inside if m - x + M < 0 => x > M + m
# laxness of l   -> inside if x > M/l + m
# laxness of M   -> inside if x > 1 + m
# laxness of inf -> inside if x > m ie inside box proper
# laxness is the inverse of the side length of the removed triangle
MAX_DEVIATION = 480
def box_truncated_swap_hsv(color: Tuple[int], source_box: List, target_box: List, laxness: List) -> Tuple[int]:
    return_color = list(color)

    # when considering a point relative to a range, abs(min - point) <= 128, abs(max - point) <= 128
    while return_color[0] < source_box[0] - 128:
        return_color[0] += 256
    while return_color[0] > source_box[1] + 128:
        return_color[0] -= 256

    # form hypothetical inner box, so that the truncated box formed has it's orthogonal faces tangent with source, eg:
    # in 1D: source min = m, inner m = m'. if x < m, x is outside, so x must deviate from m' by MAX_DEVIATION M.
    # x + M < m', or accounting for variable laxness l, M < l(m'-x), x + M/l < m', x < m' - M/l = m, m' = m + M/l
    # so deviation += (m + M/l - x)*s = (m - x)*l + M

    deviation = 0
    for i in range(3):
        deviation += max(0, (source_box[2*i] - return_color[i]) * laxness[2 * i] + MAX_DEVIATION)
        deviation += max(0, (return_color[i] - source_box[2*i + 1]) * laxness[2 * i + 1] + MAX_DEVIATION)

        if deviation > MAX_DEVIATION:  # average of 8 deviation in each dimension
            return color

    # scale coordinates to 0,1 relative to source box
    scales = tuple(weighted_average_factor(color[i], source_box[2 * i], source_box[2 * i + 1]) for i in range(3))
    # scale to absolute coordinates using target box
    return_color = [int(linear_combination(target_box[2*i], target_box[2*i+1], scales[i])) for i in range(3)]
    return_color[0] %= 256

    return tuple(return_color)


def box_swap_hsv_fuzzy(color, source_box: List, target_box: List, fuzzy_bands: List) -> Tuple[int]:
    return_color = list(color)
    swap_factor = 1

    # hue modular arithmetic must be accounted for
    # min should always be less than max for ranges, e.g. convert (200, 20) to (200, 20+256)
    while source_box[1] < source_box[0]:
        source_box[1] += 256

    while target_box[1] < target_box[0]:
        target_box[1] += 256

    # when considering a point relative to a range, abs(min - point) <= 128, abs(max - point) <= 128
    while return_color[0] < source_box[0] - 128:
        return_color[0] += 256
    while return_color[0] > source_box[1] + 128:
        return_color[0] -= 256

    # zero if value below min or above max,
    # forms trapezoid of height 1 within box_start and sloping within box_finish
    for i in range(3):
        swap_factor *= weighted_average_factor(return_color[i],
                                               source_box[2 * i] + fuzzy_bands[2 * i],
                                               source_box[2 * i])
        swap_factor *= weighted_average_factor(return_color[i],
                                               source_box[2 * i + 1] + fuzzy_bands[2 * i + 1],
                                               source_box[2 * i + 1])  # invert

    if swap_factor == 0:
        return color

    # scale coordinates to 0,1 relative to source box
    scales = tuple(weighted_average_factor(color[i], source_box[2 * i], source_box[2 * i + 1]) for i in range(3))
    # scale to absolute coordinates using target box
    target_color = tuple(linear_combination(target_box[2*i], target_box[2*i+1], scales[i]) for i in range(3))
    # swap to color
    return_color = [int(linear_combination(color[i], target_color[i], swap_factor)) for i in range(3)]
    return_color[0] %= 256

    return tuple(return_color)


FILE_PREFIX = 'image3'


# converts very unsaturated or very dark pixels only which are outside desired hue range
# which otherwise in rgb would have arbitrary or 0 hue when they shouldn't
def convert_greys(im, bounding_box, mask_array,
                  hue_min, hue_max, sat_min, val_min):
    hue_min = int(hue_min*32/45)  # convert from 360 to byte
    hue_max = int(hue_max*32/45)  # convert from 360 to byte
    sat_min = int(sat_min*64/25)  # convert from 100 to byte
    val_min = int(val_min*64/25)  # convert from 100 to byte

    im = im.convert('HSV')
    marker = Image.new(mode="L", size=im.size, color=0)

    pixels_converted = 0
    for x in range(bounding_box[0], bounding_box[1]):
        for y in range(bounding_box[2], bounding_box[3]):
            if mask_array[x][y]:
                coord = (x,y)
                hsv = im.getpixel(coord)
                marker.putpixel(coord, 127)

                if hsv[1] < sat_min or hsv[2] < val_min:
                    new_hue = hsv[0]
                    hue_mid = (hue_min + hue_max)/2
                    while new_hue < hue_mid - 128:
                        new_hue += 256
                    while new_hue > hue_mid + 128:
                        new_hue -= 256

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


# construct a compound mask:
# everything indicated by the initial mask AND in source box is black
# in source box but not in initial mask = red
# in initial mask but not source box = green
def create_composite_mask(im, mask, byte_source_box: List[int],
                          box_not_mask_color=(255, 0, 0), mask_not_box_color=(0, 255, 0)):

    mask = mask.convert('RGB')
    im = im.convert('HSV')
    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)
            isInMask = mask.getpixel(coord)[0] < 128
            isInBox = True

            imageColor = list(im.getpixel(coord))
            # when considering a point relative to a range, abs(min - point) <= 128, abs(max - point) <= 128
            while imageColor[0] < byte_source_box[0] - 128:
                imageColor[0] += 256
            while imageColor[0] > byte_source_box[1] + 128:
                imageColor[0] -= 256

            for i in range(3):
                if imageColor[i] < byte_source_box[2 * i] or byte_source_box[2 * i + 1] < imageColor[i]:
                    isInBox = False
                    break

            if not isInMask and isInBox:
                mask.putpixel(coord, box_not_mask_color)
            if isInMask and not isInBox:
                mask.putpixel(coord, mask_not_box_color)
    mask.save(FILE_PREFIX + 'mask_compound.png')


def create_spectra():
    im = Image.new(mode="HSV", size=(256, 256), color=(0, 0, 0))
    for x in range(im.size[0]):
        for y in range(im.size[1]//2):
            im.putpixel((x, y), (x, 255, 255))
        for y in range(im.size[1]//2, im.size[1]):
            im.putpixel((x, y), (x, 255, 510 - 2*y))
    im = im.convert('RGB')
    im.save('spectrum.png')

    im = im.convert('HSV')
    im = map_pixels(im, lambda c: box_swap_hsv(c,
                                               [100, 156, 0, 255, 0, 255],
                                               [95, 161, -1, 256, -1, 256]),
                    [0, 255, 64, 255])
    im = im.convert('RGB')
    im.save('spectrum2.png')

    im = Image.open('spectrum.png')
    im = im.convert('HSV')
    im = map_pixels(im, lambda c: box_swap_hsv(c,
                                               [100, 156, 0, 255, 0, 255],
                                               [95, 161, -1, 256, -1, 256]),
                    [0, 255, 64, 255])
    im = im.convert('RGB')
    im.save('spectrum3.png')


def create_color_permutations():
    im = Image.open(FILE_PREFIX + '.png')
    im = linear_map_colors(im, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    im.save(FILE_PREFIX + '_rbg.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = linear_map_colors(im, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    im.save(FILE_PREFIX + '_grb.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = linear_map_colors(im, [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    im.save(FILE_PREFIX + '_brg.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = linear_map_colors(im, [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    im.save(FILE_PREFIX + '_gbr.png')

    im = Image.open(FILE_PREFIX + '.png')
    im = linear_map_colors(im, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    im.save(FILE_PREFIX + '_bgr.png')


def create_hue_rotations(rotation_amount=24):
    for i in range(rotation_amount, 256, rotation_amount):
        im = Image.open(FILE_PREFIX + '.png')
        im = im.convert('HSV')
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                hsv = im.getpixel((x,y))
                im.putpixel((x,y), ((hsv[0] + i) % 256, hsv[1], hsv[2]))
        im = im.convert('RGB')
        im.save(FILE_PREFIX + f'_rot{i:03d}.png')


def run():
    source_box = [190, 350, 0, 75, 15, 100]
    byte_source_box = convert_hsv_box_to_bytes(source_box)

    try:
        mask = Image.open((FILE_PREFIX + 'mask2.png'))
        print('using alternate mask')
    except FileNotFoundError:
        mask = Image.open((FILE_PREFIX + 'mask.png'))
    mask = mask.convert('RGB')

    try:
        im = Image.open(FILE_PREFIX + 'a.png')  # alternate or edited image
        print('using alternate image')
    except FileNotFoundError:
        im = Image.open(FILE_PREFIX + '.png')
    im = im.convert('HSV')

    # create_composite_mask(im, mask, byte_source_box)
    # exit(0)

    bounding_box = [mask.size[0], 0, mask.size[1], 0]
    mask_array = [[False for y in range(mask.size[1])] for x in range(mask.size[0])]
    coord_list = []
    hsv_list = []

    combine_mask_with_source_box = True
    if combine_mask_with_source_box:
        for x in range(mask.size[0]):
            for y in range(mask.size[1]):
                coord = (x, y)
                isInMask = mask.getpixel(coord)[0] < 128
                isInBox = True

                imageColor = list(im.getpixel(coord))
                # when considering a point relative to a range, abs(min - point) <= 128, abs(max - point) <= 128
                while imageColor[0] < byte_source_box[0] - 128:
                    imageColor[0] += 256
                while imageColor[0] > byte_source_box[1] + 128:
                    imageColor[0] -= 256

                for i in range(3):
                    if imageColor[i] < byte_source_box[2 * i] or byte_source_box[2 * i + 1] < imageColor[i]:
                        isInBox = False
                        break

                if isInMask and isInBox:
                    mask_array[x][y] = True
                    coord_list.append(coord)
                    hsv_list.append(tuple(imageColor))
                    if bounding_box[0] > x:
                        bounding_box[0] = x
                    if bounding_box[1] < x:
                        bounding_box[1] = x
                    if bounding_box[2] > y:
                        bounding_box[2] = y
                    if bounding_box[3] < y:
                        bounding_box[3] = y
    else:
        for x in range(mask.size[0]):
            for y in range(mask.size[1]):
                coord = (x, y)
                if mask.getpixel(coord)[0] < 128:
                    imageColor = list(im.getpixel(coord))
                    mask_array[x][y] = True
                    coord_list.append(coord)
                    hsv_list.append(tuple(imageColor))
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

    bounding_box[1] += 1
    bounding_box[3] += 1

    if not combine_mask_with_source_box:
        convert_greys(im, bounding_box, mask_array, source_box[0], source_box[1], 9, 9)
        exit(0)

    target_boxes = []

    hue_range = int((source_box[1] - source_box[0])/1)

    box_labels = ['black', 'grey', 'white']
    target_boxes.append([source_box[0],source_box[1],0,10,0,35])
    target_boxes.append([source_box[0],source_box[1],0,10,30,70])
    target_boxes.append([source_box[0],source_box[1],0,10,65,100])

    def convert_range(source_lo, source_hi, target_mid):
        radius = (source_hi - source_lo) // 2
        target_range = [target_mid - radius, target_mid + radius]
        overflow = target_range[1] - 100
        if overflow > 0:
            target_range[1] = 100
            target_range[0] += overflow  # need to move lo point up to move midpoint since hi is maxed
        if target_range[0] < 0:
            target_range[1] += target_range[0]   # need to move hi point down to move midpoint since lo is minimal
            target_range[0] = 0
        target_range[0] = min(90, target_range[0])
        target_range[1] = max(10, target_range[1])
        return target_range

    hi_sat_range = convert_range(source_box[2], source_box[3], 70)
    lo_sat_range = convert_range(source_box[2], source_box[3], 30)

    hi_val_range = convert_range(source_box[4], source_box[5], 70)
    lo_val_range = convert_range(source_box[4], source_box[5], 30)

    num_hues = 18
    # even_spaced_hues = [hue for hue in range(0, 360, 360 // num_hues)]
    uneven_spaced_hues = [hue for hue in range(-60, 120, 270 // num_hues)]
    uneven_spaced_hues.extend([hue for hue in range(120, 300, 540 // num_hues)])

    for hue in uneven_spaced_hues:
        hue_start = hue - hue_range//2
        hue_end = hue_start + hue_range

        box_labels.append('sameSV')
        target_boxes.append([hue_start, hue_end, source_box[2], source_box[3], source_box[4], source_box[5]])

        box_labels.append('bright')
        target_boxes.append([hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], hi_val_range[0], hi_val_range[1]])

        box_labels.append('light')
        target_boxes.append([hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], hi_val_range[0], hi_val_range[1]])

        box_labels.append('dark')
        target_boxes.append([hue_start, hue_end, hi_sat_range[0], hi_sat_range[1], lo_val_range[0], lo_val_range[1]])

        box_labels.append('dull')
        target_boxes.append([hue_start, hue_end, lo_sat_range[0], lo_sat_range[1], lo_val_range[0], lo_val_range[1]])

    for target_box, label in zip(target_boxes, box_labels):
        try:
            im = Image.open(FILE_PREFIX + 'a.png')  # alternate or edited image
        except FileNotFoundError:
            im = Image.open(FILE_PREFIX + '.png')
        im = im.convert('HSV')

        filename = FILE_PREFIX + f'_h{target_box[0]%360:03d}_{target_box[1]%360:03d}' \
                                 f'_s{target_box[2]:03d}_{target_box[3]:03d}' \
                                 f'_v{target_box[4]:03d}_{target_box[5]:03d}_{label}.png'
        byte_target_box = convert_hsv_box_to_bytes(target_box)

        # precompute conversions for each possible byte for HSV
        converted_hsv = [[],[],[]]
        for b in range(256):
            for i in range(3):
                # scale coordinates to 0,1 relative to source box
                scale = weighted_average_factor(b, byte_source_box[2 * i], byte_source_box[2 * i + 1])
                # scale to absolute coordinates using target box
                converted = int(linear_combination(byte_target_box[2 * i], byte_target_box[2 * i + 1], scale)) % 256
                converted_hsv[i].append(converted)

        for coord, hsv in zip(coord_list, hsv_list):
            im.putpixel(coord, (converted_hsv[0][hsv[0]], converted_hsv[1][hsv[1]], converted_hsv[2][hsv[2]]))

        im = im.convert('RGB')
        im.save(filename)
        print(filename)


run()
# cProfile.run('run()', sort='tottime')

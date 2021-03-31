from typing import Tuple, List, Callable


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


# returns hue which is congruent mod m which is nearest to center of min and max
def hue_nearest_bands(old_hue, hue_min, hue_max, m=256):
    hue_mid = (hue_min + hue_max)/2
    while old_hue < hue_mid - m/2:
        old_hue += m
    while old_hue > hue_mid + m/2:
        old_hue -= m
    return old_hue


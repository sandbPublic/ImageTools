import math
import os

from PIL import Image, ImageDraw
from typing import Tuple, List, AnyStr
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


# see twocolor
@dataclass
class RGV:
    r: int  # red
    g: int  # purple
    v: int

    def to_string(self):
        return f'V{self.v:03d}_R{self.r:03d}_G{self.g:03d}'


class Neighbor:
    def __init__(self, i, j):
        self.i: int = i
        self.j: int = j
        # division is more expensive then multiplication, so precompute inverse and multiply
        self.inverse_distance = 1/math.sqrt(i * i + j * j)


def generate_neighbors(manhattan_radius):
    neighbors = []
    for i in range(-manhattan_radius, manhattan_radius + 1):
        for j in range(-manhattan_radius, manhattan_radius + 1):
            if i == 0 and j == 0:
                continue

            neighbors.append(Neighbor(i, j))

    return neighbors


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

    def set_pixel(hsv):
        scatter_hs.putpixel((hsv[0], hsv[1]), (hsv[0], hsv[1], 255))
        scatter_sv.putpixel((hsv[1], hsv[2]), (0, hsv[1], hsv[2]))
        scatter_hv.putpixel((hsv[0], hsv[2]), (hsv[0], 255, hsv[2]))

    if hsv_list is None:
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                set_pixel(im.getpixel((x, y)))
    else:
        for hsv in hsv_list:
            set_pixel(hsv)

    scatter_hs = scatter_hs.convert('RGB')
    scatter_hs.save(FILE_PREFIX + '_scatter_hs.png')
    scatter_sv = scatter_sv.convert('RGB')
    scatter_sv.save(FILE_PREFIX + '_scatter_sv.png')
    scatter_hv = scatter_hv.convert('RGB')
    scatter_hv.save(FILE_PREFIX + '_scatter_hv.png')


def create_hsv_masks(im):
    im = im.convert('HSV')

    h_mask = im.copy()
    s_mask = im.copy()
    v_mask = im.copy()

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            coord = (x, y)
            hsv = im.getpixel((x, y))
            h_mask.putpixel(coord, (hsv[0], 255, 255))
            s_mask.putpixel(coord, (0, hsv[1], 255))
            v_mask.putpixel(coord, (0, 0, hsv[2]))


    h_mask = h_mask.convert('RGB')
    h_mask.save(FILE_PREFIX + '_hue_mask.png')
    s_mask = s_mask.convert('RGB')
    s_mask.save(FILE_PREFIX + '_sat_mask.png')
    v_mask = v_mask.convert('RGB')
    v_mask.save(FILE_PREFIX + '_val_mask.png')


include_color = (0, 0, 0)
exclude_color = (255, 255, 255)
conditional_color = (0, 191, 0)
# for marking pixels in conditional zone
in_source_color = (0, 255, 0)
out_source_color = (255, 0, 0)

def color_is_in(c):
    return c == include_color or c == in_source_color

def nxor(a, b):
    return (a and b) or (not a and not b)

def create_suggested_mask(c_mask,
                          suggested_radius=2,
                          suggested_threshold=0.75):
    s_mask = c_mask.copy()
    cached_inverse_roots = [1.0 / math.sqrt(x) if x > 0 else 0 for x in
                            range(suggested_radius * suggested_radius * 2 + 1)]
    agreement_factor = 0
    for i in range(-suggested_radius, suggested_radius + 1):
        for j in range(-suggested_radius, suggested_radius + 1):
            agreement_factor += cached_inverse_roots[i * i + j * j]
    agreement_factor = 1 / agreement_factor  # will be used for division

    pixels_suggested = 0
    for x in range(suggested_radius, c_mask.size[0] - suggested_radius):
        for y in range(suggested_radius, c_mask.size[1] - suggested_radius):
            coord = (x, y)

            c_mask_rgb = c_mask.getpixel(coord)
            if c_mask_rgb == include_color or c_mask_rgb == exclude_color:
                continue

            agreement = 0
            disagreement = 0
            for i in range(-suggested_radius, suggested_radius + 1):
                for j in range(-suggested_radius, suggested_radius + 1):
                    neighbor = c_mask.getpixel((x + i, y + j))
                    if nxor(color_is_in(neighbor), color_is_in(c_mask_rgb)):
                        agreement += cached_inverse_roots[i * i + j * j]
                    else:
                        disagreement += cached_inverse_roots[i * i + j * j]

            if disagreement * agreement_factor >= suggested_threshold:
                pixels_suggested += 1
                if color_is_in(c_mask_rgb):
                    s_mask.putpixel(coord, exclude_color)
                else:
                    s_mask.putpixel(coord, include_color)
            else:
                s_mask.putpixel(coord, conditional_color)

    print(f'pixels_suggested {pixels_suggested}')
    s_mask.save(FILE_PREFIX + 'mask_suggested.png')

# Construct a compound mask:
# Included or excluded in the initial mask remains the same,
# conditional in initial mask and in source box = green,
# conditional in initial mask but not in source box = red.
# Also create a suggested mask based on replacing outliers
# in conditional area with explicit exclusion/inclusion.
def create_composite_mask(im,
                          mask,
                          byte_source_box: List[int]):
    mask = mask.convert('RGB')
    im = im.convert('HSV')
    c_mask = Image.new(mode='RGB', size=mask.size, color=(255, 255, 255))

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if mask_rgb == include_color or mask_rgb == exclude_color:
                c_mask.putpixel(coord, mask_rgb)
                continue

            image_hsv = im.getpixel(coord)
            image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])

            if byte_source_box[0] <= image_hue <= byte_source_box[1] and \
                    byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and \
                    byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]:
                c_mask.putpixel(coord, in_source_color)
            else:
                c_mask.putpixel(coord, out_source_color)

    c_mask.save(FILE_PREFIX + 'mask_compound.png')
    # create_suggested_mask(c_mask)


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


def preprocess_image_and_mask_hsv(im, mask, byte_source_box=None):
    coord_list = []
    hsv_list = []
    hue_counts = [0] * 256
    sat_counts = [0] * 256
    val_sum = 0

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if mask_rgb == exclude_color:
                continue

            image_hsv: Tuple[int, int, int] = im.getpixel(coord)

            # now either included or conditional 
            # if no source box provided, treat as if it is in i.e. OK
            if mask_rgb == conditional_color and byte_source_box is not None:
                image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])
                if not (byte_source_box[0] <= image_hue <= byte_source_box[1] and
                        byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and
                        byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]):
                    continue  # conditional mask region and outside source box (failed)

            coord_list.append(coord)
            hsv_list.append(image_hsv)

            # dark and desat pixels don't contribute as much to hue, weight appropriately
            hue_counts[image_hsv[0]] += math.sqrt(image_hsv[1]*image_hsv[2])
            # dark pixels don't contribute as much to sat, weight appropriately
            sat_counts[image_hsv[1]] += image_hsv[2]
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
                  sum([i * count for i, count in enumerate(sat_counts)]) // sum(sat_counts),
                  val_sum // included_pixel_count)
    print(f'average hsv: {avg_hsv.to_string_from_bytes()}')

    return coord_list, hsv_list, avg_hsv


def preprocess_image_and_mask_rgv(im, mask, byte_source_box=None):
    coord_list = []
    rgv_list = []
    red_sum = 0
    green_sum = 0
    val_sum = 0

    for x in range(mask.size[0]):
        for y in range(mask.size[1]):
            coord = (x, y)

            mask_rgb = mask.getpixel(coord)
            if mask_rgb == exclude_color:
                continue

            image_hsv: Tuple[int, int, int] = im.getpixel(coord)

            # now either included or conditional
            # if no source box provided, treat as if it is in i.e. OK
            if mask_rgb == conditional_color and byte_source_box is not None:
                image_hue = cm.hue_nearest_bands(image_hsv[0], byte_source_box[0], byte_source_box[1])
                if not (byte_source_box[0] <= image_hue <= byte_source_box[1] and
                        byte_source_box[2] <= image_hsv[1] <= byte_source_box[3] and
                        byte_source_box[4] <= image_hsv[2] <= byte_source_box[5]):
                    continue  # conditional mask region and outside source box (failed)

            r, g = cm.HS_to_twocolor(image_hsv[0], image_hsv[1])
            image_rgv = RGV(r, g, image_hsv[2])
            coord_list.append(coord)
            rgv_list.append(image_rgv)

            val_sum += image_rgv.v

            # dark pixels don't contribute as much to twocolor, weight appropriately
            # normalize to grey = 0 to do this, then translate back to unsigned
            red_sum += (image_rgv.r - 128) * image_rgv.v
            green_sum += (image_rgv.g - 128) * image_rgv.v


    included_pixel_count = len(coord_list)
    print(f'mask_list size {included_pixel_count}')

    avg_rgv = RGV(128 + (red_sum // val_sum),
                  128 + (green_sum // val_sum),
                  val_sum // included_pixel_count)
    print(f'average rgv: {avg_rgv.r} {avg_rgv.g} {avg_rgv.v}')

    return coord_list, rgv_list, avg_rgv


num_hues = 18
# even_spaced_hues = [hue for hue in range(0, 360, 360 // num_hues)]
uneven_spaced_hues = [hue for hue in range(-60, 120, 270 // num_hues)]
uneven_spaced_hues.extend([hue for hue in range(120, 300, 540 // num_hues)])


def create_mapping_guide(hsv_mappings: List[HSV], column_size=16):
    im = Image.new(mode="HSV", size=(256, 6 * column_size), color=(0, 0, 0))
    drawer = ImageDraw.Draw(im)

    for x in range(256):
        drawer.line([x, 0 * column_size, x, 1 * column_size - 1], (x, 255, 255))
        drawer.line([x, 1 * column_size, x, 2 * column_size - 1], (hsv_mappings[x].h, 255, 255))
        drawer.line([x, 2 * column_size, x, 3 * column_size - 1], (0, x, 255))
        drawer.line([x, 3 * column_size, x, 4 * column_size - 1], (0, hsv_mappings[x].s, 255))
        drawer.line([x, 4 * column_size, x, 5 * column_size - 1], (0, 0, x))
        drawer.line([x, 5 * column_size, x, 6 * column_size - 1], (0, 0, hsv_mappings[x].v))

    im = im.convert('RGB')
    im.save(f'conversion_guide.png')


# Precompute conversions for each possible byte for H, S, and V.
# hue_contraction = K: new(h) = h + t - a - K*(h - a)
# hue_contraction of 1 will contract to a point at target_hsv.h, 0 does nothing.
# Negative values will expand and can create multicolor or rainbow effects.
# S and V multiply (or multiply complement) to change average to target
# Preserve_dynamics from 0 to 1 will preserve dark/light and saturated/unsaturated areas,
# but will not reach target average
def average_to_target_mappings(avg_hsv: HSV,
                               target_hsv: HSV,
                               hue_contraction=0.0,
                               preserve_sats=0.0,
                               preserve_vals=0.0):
    hue_delta = cm.nearest_congruent(target_hsv.h - avg_hsv.h)
    return [HSV(int((b + hue_delta - hue_contraction * (b - avg_hsv.h))) % 256,
                cm.hybrid_mult(b, avg_hsv.s, target_hsv.s, preserve_sats),
                cm.hybrid_mult(b, avg_hsv.v, target_hsv.v, preserve_vals))
            for b in range(256)]


# maps RG -> HS, and V -> V:
# a [256][256] -> HS and [256] -> V array
def average_to_target_mappings_rgv(avg_rgv: RGV,
                                   target_rgv: RGV,
                                   preserve_vals=1.0):
    val_conversion = [cm.hybrid_mult(b, avg_rgv.v, target_rgv.v, preserve_vals) for b in range(256)]
    RGtoHSconversion = []
    for r in range(256):
        RGtoHSconversion.append([])
        new_r = cm.hybrid_mult(r, avg_rgv.r, target_rgv.r, 0.5)
        for g in range(256):
            new_g = cm.hybrid_mult(g, avg_rgv.g, target_rgv.g, 0.5)
            RGtoHSconversion[r].append(cm.twocolor_to_HS(new_r, new_g))

    return RGtoHSconversion, val_conversion


def create_color_swap(im, hsv_mappings, coord_list, hsv_list, filename):
    im = im.convert('HSV')

    for coord, hsv in zip(coord_list, hsv_list):
        im.putpixel(coord, (hsv_mappings[hsv[0]].h, hsv_mappings[hsv[1]].s, hsv_mappings[hsv[2]].v))

    im = im.convert('RGB')
    im.save(filename)
    print(filename)


def create_color_swap_rgv(im, rg_mapping, v_mapping, coord_list, rgv_list, filename: AnyStr):
    im = im.convert('HSV')

    for coord, rgv in zip(coord_list, rgv_list):
        im.putpixel(coord, (rg_mapping[rgv.r][rgv.g][0], rg_mapping[rgv.r][rgv.g][1], v_mapping[rgv.v]))

    im = im.convert('RGB')
    im.save(filename)
    print(filename)


def create_edge_detection(im, buffer=1):
    im = im.convert('RGB')
    edges = Image.new(mode='RGB', size=im.size, color=(0, 0, 0))

    cached_inverse_roots = [1.0/math.sqrt(x) if x > 0 else 0 for x in range(buffer * buffer * 2 + 1)]
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

                    neighbor = im.getpixel((x + i, y + j))
                    for c in range(3):
                        total_contrast[c] += abs(neighbor[c] - current_pixel[c]) * cached_inverse_roots[square_dist]

            unscaled_values[x][y] = total_contrast

            for c in total_contrast:
                if greatest_contrast < c:
                    greatest_contrast = c

    print("Scaling edges...")
    for x in range(buffer, im.size[0] - buffer):
        for y in range(buffer, im.size[1] - buffer):
            edges.putpixel((x, y), tuple([int(c * 255 / greatest_contrast) for c in unscaled_values[x][y]]))

    edges.save(FILE_PREFIX + "_edges.png")


def create_image_diff(imA, imB):
    print('creating image diff')
    diff = Image.new(mode='RGB', size=imA.size, color=(128, 128, 128))

    for x in range(imA.size[0]):
        for y in range(imB.size[1]):
            coord = (x, y)

            pixelA = imA.getpixel(coord)
            pixelB = imB.getpixel(coord)

            pixelDiff = [pixelA[i] - pixelB[i] for i in range(3)]
            if pixelDiff[0] == 0 and pixelDiff[1] == 0 and pixelDiff[2] == 0:
                continue

            for i in range(3):
                if pixelDiff[i] < 0:
                    pixelDiff[i] = 64 + pixelDiff[i] // 4
                elif pixelDiff[i] > 0:
                    pixelDiff[i] = 192 + pixelDiff[i] // 4

            diff.putpixel(coord, tuple(pixelDiff))

    diff.save('diff.png')


# todo get filenames from input()
def run(use_hsv=True):
    im = None
    directory = ""
    for filename in os.listdir():
        if filename.startswith('_' + FILE_PREFIX) and filename.endswith('.png'):
            print('using', filename)
            im = Image.open(filename)
            directory = f'{FILE_PREFIX}alts{filename[1 + len(FILE_PREFIX):-4]}'
            break
    else:
        print('file not found for', FILE_PREFIX)
        exit(1)

    im = im.convert('HSV')

    # create_histograms(im)
    # create_scatterplots(im)
    # create_hsv_masks(im)
    # create_edge_detection(im, 2)
    # blur = blur_low_contrast_only(im)
    # create_image_diff(im, blur)
    # create_val_to_color(im, 255, 128, -64, 448)
    # exit(0)

    mask = Image.open((FILE_PREFIX + 'mask.png'))
    mask = mask.convert('RGB')

    source_box = [240, 350, 0, 80, 0, 100]
    byte_source_box = cm.convert_hsv_box_to_bytes(source_box)

    create_composite_mask(im, mask, byte_source_box)
    #exit(0)

    # if existing sat or val is low/high enough to serve as such, create a new "mid" value
    def create_SV_low_mid_high(x, SVscale=0.5, required_range=32):
        mid = x
        low = mid * SVscale
        hi = 255 - (255 - mid) * SVscale

        if low < required_range:
            low = mid
            mid = low / SVscale
            hi = 255 - (255 - mid) * SVscale

        if hi > 255 - required_range:
            hi = mid
            mid = 255 - (255 - hi) / SVscale
            low = mid * SVscale

        return low, mid, hi

    if not os.path.exists(directory):
        os.makedirs(directory)

    if use_hsv:
        coord_list, hsv_list, avg_hsv = preprocess_image_and_mask_hsv(im, mask, byte_source_box)

        # create_histograms(im, hsv_list)
        # create_scatterplots(im, hsv_list)
        # exit(0)

        # convert_greys(im, coord_list, hsv_list, source_box[0], source_box[1], 9, 9)
        # exit(0)

        satLMH = create_SV_low_mid_high(avg_hsv.s)
        valLMH = create_SV_low_mid_high(avg_hsv.v)

        target_sats = [satLMH[0], satLMH[0], satLMH[2], satLMH[2], satLMH[1]]
        target_vals = [valLMH[0], valLMH[2], valLMH[0], valLMH[2], valLMH[1]]

        all_targets = [HSV(255, 0, 48), HSV(255, 0, 224)]
        all_special_params = [[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]]  # do not preserve sat for black/white

        for hue in uneven_spaced_hues:
            for sat, val in zip(target_sats, target_vals):
                all_targets.append(HSV(int(hue * 32/45), sat, val))
                all_special_params.append([1.0, 0.5, 0.5])

        for target, special_params in zip(all_targets, all_special_params):
            create_color_swap(im.copy(),
                              average_to_target_mappings(avg_hsv,
                                                         target,
                                                         special_params[0],
                                                         special_params[1],
                                                         special_params[2]),
                              coord_list,
                              hsv_list,
                              f'{directory}/{FILE_PREFIX}_{target.to_string_from_bytes()}.png')
    else:
        coord_list, rgv_list, avg_rgv = preprocess_image_and_mask_rgv(im, mask, byte_source_box)

        valLMH = create_SV_low_mid_high(avg_rgv.v)
        all_targets = [RGV(128,128,0), RGV(128,128,255)]
        for i, v in enumerate(valLMH):
            num_rows = 3 #+ 2*i  # larger squares farther from black
            targetRGs = [int(cm.linear_combination(0, 255, cm.weighted_average_factor(x, 0, num_rows-1))) for x in range(num_rows)]
            for r in targetRGs:
                for g in targetRGs:
                    all_targets.append(RGV(r, g, int(v)))

        for target in all_targets:
            rg_mapping, v_mapping = average_to_target_mappings_rgv(avg_rgv, target, 1.0)

            create_color_swap_rgv(im.copy(),
                                  rg_mapping,
                                  v_mapping,
                                  coord_list,
                                  rgv_list,
                                  f'{directory}/{FILE_PREFIX}_{target.to_string()}.png')


if __name__ == '__main__':
    run(False)
    # cProfile.run('run()', sort='tottime')

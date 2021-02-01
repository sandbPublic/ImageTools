from PIL import Image
import random
import cProfile
from typing import List

X_MAX = 540
Y_MAX = 360
C_MAX_FLOAT = 255.99


class VoronoiPoint:
    def __init__(self, x, y, c, scale=1):
        self.x = x
        self.y = y
        self.color = c
        self.scale = scale
        self.area = 0

    def scaled_square_dist(self, x, y):
        dx = self.x - x
        dy = self.y - y
        return self.scale*(dx*dx + dy*dy)


def color_by_nearest(x, y, v_points) -> List[float]:
    if len(v_points) < 1:
        return [0, 0, 0]

    min_dist = v_points[0].scaled_square_dist(x, y)
    color = v_points[0].color
    for p in v_points:
        if min_dist > p.scaled_square_dist(x, y):
            min_dist = p.scaled_square_dist(x, y)
            color = p.color

    return [c for c in color]  # make copy


def color_by_blend(x, y, v_points) -> List[float]:
    if len(v_points) < 1:
        return [0, 0, 0]

    total_scaled_square_dist = 0
    total_color = [0, 0, 0]
    for p in v_points:
        scale = 1/(1+p.scaled_square_dist(x, y)**3)
        total_scaled_square_dist += scale
        for i in range(3):
            total_color[i] += p.color[i] * scale

    for i in range(3):
        total_color[i] /= total_scaled_square_dist

    return total_color


def color_by_probability(x, y, v_points) -> List[float]:
    if len(v_points) < 1:
        return [0, 0, 0]

    prob_weits = []
    running_weit = 0

    for p in v_points:
        running_weit += 1/(1+p.scaled_square_dist(x, y)**3)
        prob_weits.append(running_weit)

    roll = random.uniform(0, running_weit)
    for p, w in zip(v_points, prob_weits):
        if roll <= w:
            return [c for c in p.color]  # make copy

    return [0, 0, 0]


# expand range for colors so that there is a point with value of 0 and a point with value 255 for each of R,G,B.
# however this does not prevent eventual 2-hued images, from white-black, red-cyan, green-violet, blue-yellow
# scale from [0, 1] dictates the extent of the effect
def extend_colors(v_points, scale=1):
    for i in range(3):
        min_color = v_points[0].color[i]
        max_color = min_color
        for p in v_points:
            if max_color < p.color[i]:
                max_color = p.color[i]
            if min_color > p.color[i]:
                min_color = p.color[i]

        color_range = max_color - min_color
        if color_range == 0:
            continue

        color_scale = 1 + (C_MAX_FLOAT/color_range - 1)*scale
        min_after_scale = min_color*(1-scale)

        for p in v_points:
            p.color[i] -= min_color
            p.color[i] *= color_scale
            p.color[i] += min_after_scale


def meta_v_points(num_points, old_v_points, color_method=color_by_nearest, variance=0.0):
    new_points = [VoronoiPoint(random.uniform(0, X_MAX), random.uniform(0, Y_MAX), [0,0,0]) for _ in range(num_points)]
    for new_point in new_points:
        new_point.color = [min(max(0.0, c + random.uniform(-variance, variance)), C_MAX_FLOAT)
                           for c in color_method(new_point.x, new_point.y, old_v_points)]
    return new_points


def create_image(v_points, im, color_method=color_by_nearest):
    pixels = im.load()

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            pixels[x, y] = tuple(int(i) for i in color_method(x, y, v_points))


# color every other pixel in every other line normally
# then for blank pixels, copy based on average of opposite neibors
def quick_create(v_points, im, color_method=color_by_nearest):
    pixels = im.load()

    # every even pixel, every even row
    for x in range(0, im.size[0], 2):
        for y in range(0, im.size[1], 2):
            pixels[x, y] = tuple(int(i) for i in color_method(x, y, v_points))

    # last pixel of every even row if last pixel index is odd
    if im.size[0] % 2 == 0:
        for y in range(0, im.size[1], 2):
            pixels[im.size[0]-1, y] = tuple(int(i) for i in color_method(im.size[0]-1, y, v_points))

    # every even pixel of last row if last row index is odd
    if im.size[1] % 2 == 0:
        for x in range(0, im.size[0], 2):
            pixels[x, im.size[1]-1] = tuple(int(i) for i in color_method(x, im.size[1]-1, v_points))

    # every odd pixel of every even row
    for x in range(1, im.size[0]-1, 2):
        for y in range(0, im.size[1]-1, 2):
            pixels[x, y] = tuple(int((pixels[x-1, y][i] + pixels[x+1, y][i])/2) for i in range(3))

    # every pixel of every odd row
    for x in range(im.size[0]-1):
        for y in range(1, im.size[1]-1, 2):
            pixels[x, y] = tuple(int((pixels[x-1, y][i] + pixels[x+1, y][i])/2) for i in range(3))


def mark_points(v_points, pixels):
    for p in v_points:
        ix = int(p.x)
        iy = int(p.y)
        for x in range(max(ix-1, 0), min(ix+2, X_MAX)):
            for y in range(max(iy-1, 0), min(iy+2, Y_MAX)):
                pixels[x, y] = tuple((c + 128) % 256 for c in pixels[x, y])


def image_to_voronoi(im, num_points):
    v_points = [VoronoiPoint(random.uniform(0, im.size[0]), random.uniform(0, im.size[1]), [0,0,0]) for _ in range(num_points)]

    for x in range(im.size[0]):
        for y in range(im.size[1]):
            nearest_point = v_points[0]
            min_dist = nearest_point.scaled_square_dist(x, y)
            for p in v_points[1:]:
                if min_dist > p.scaled_square_dist(x, y):
                    min_dist = p.scaled_square_dist(x, y)
                    nearest_point = p

            for i, c in enumerate(im.getpixel((x, y))):
                nearest_point.color[i] += c
            nearest_point.area += 1

    for vp in v_points:
        if vp.area > 0:
            for i in range(3):
                vp.color[i] /= vp.area

    return v_points


def demo():
    im = Image.new(mode="RGB", size=(X_MAX, Y_MAX), color=(0, 0, 0))
    pixels = im.load()
    image_num = 0

    def save_image():
        nonlocal image_num
        im.save(f'iter{image_num:03d}.png')
        print(f'saved iter{image_num:03d}.png')
        image_num += 1

    for _ in range(3):
        v_points = [VoronoiPoint(random.uniform(0, X_MAX), random.uniform(0, Y_MAX),
                                 [random.uniform(0, C_MAX_FLOAT), random.uniform(0, C_MAX_FLOAT),
                                  random.uniform(0, C_MAX_FLOAT)])
                    for _ in range(8)]
        v_points = meta_v_points(len(v_points) * 3, v_points, variance=10.0)
        for p in v_points:
            p.scale += random.random()
        extend_colors(v_points)
        create_image(v_points, pixels, color_by_nearest)
        save_image()
        # mark_points(v_points, pixels)
        # save_image()
        # create_image(v_points, pixels, color_by_blend)
        # save_image()
        create_image(v_points, pixels, color_by_probability)
        save_image()


def run(file_prefix):
    source = Image.open((file_prefix + '.png'))
    v_points = image_to_voronoi(source, 256)
    for p in v_points:
        p.scale += random.random()
    extend_colors(v_points)

    im = Image.new(mode="RGB", size=source.size)
    create_image(v_points, im, color_by_nearest)
    im.save(file_prefix + f'_voronoized{len(v_points)}.png')


#run('imageA')
cProfile.run('run(\'imageA\')', sort='tottime')

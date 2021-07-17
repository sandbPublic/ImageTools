from PIL import Image, ImageDraw
import colormanip as cm


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


def create_twocolor(val=255):
    im = Image.new(mode="HSV", size=(256, 256), color=(0, 0, 0))
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            hue, sat = cm.twocolor_to_HS(x,y)
            im.putpixel((x, y), (hue, sat, val))
    im = im.convert('RGB')
    im.save('twocolor.png')


def create_lab(L=255):
    im = Image.new(mode="LAB", size=(256, 256), color=(0, 0, 0))
    for a in range(im.size[0]):
        for b in range(im.size[1]):
            im.putpixel((a, b), (L, a, b))
    im.save(f'Lab{L:03d}.tiff')


if __name__ == '__main__':
    create_lab(0)
    create_lab(64)
    create_lab(128)
    create_lab(192)
    create_lab(255)
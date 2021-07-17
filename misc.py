from PIL import Image, ImageFilter
import os
import colormanip as cm

def blur(im, radius=1):
    im = im.convert('RGB')
    im = im.filter(ImageFilter.GaussianBlur(radius))
    return im


def unmoire(im, radius=1, percent=200, threshold=3):
    im = im.convert('RGB')
    im = im.filter(ImageFilter.GaussianBlur(radius))
    return im.filter(ImageFilter.UnsharpMask(radius, percent, threshold))


def unmoire_and_greyscale(im, radius=1, percent=200, threshold=3):
    im = im.convert('RGB')
    im = im.filter(ImageFilter.GaussianBlur(radius))
    im = im.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
    return im.convert('L')


# create saturation, 0 at val = 0 or 255, peak_sat at val = peak_sat_point, linear interpolation in between.
# same for hue but custom endpoints
def create_val_to_color(im, dark_hue, light_hue, peak_sat=255, peak_sat_point=128, coord_list=None):
    im = im.convert('HSV')

    val_to_sat = []
    val_to_hue = []
    for val in range(256):
        factor = cm.weighted_average_factor(val, peak_sat_point, 0 if val <= peak_sat_point else 255)
        val_to_sat.append(int(cm.linear_combination(peak_sat, 0, factor)))
        val_to_hue.append(int(cm.nearest_congruent(cm.linear_combination(light_hue, dark_hue, val/255))))

    def apply_to_coord(coord):
        val = im.getpixel(coord)[2]
        im.putpixel(coord, (val_to_hue[val], val_to_sat[val], val))

    if coord_list is None:
        for x in range(im.size[0]):
            for y in range(im.size[1]):
                apply_to_coord((x, y))
    else:
        for coord in coord_list:
            apply_to_coord(coord)

    im = im.convert('RGB')
    return im


def filter_all(directory, subdirectory, additional_params=''):
    if not os.path.exists(f'{directory}/{subdirectory}'):
        os.makedirs(f'{directory}/{subdirectory}')

    print(subdirectory, directory)
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            im = Image.open(directory + "/" + filename)
            local_dict = locals()
            command = f'im = {subdirectory}(im{additional_params})'
            exec(command, globals(), local_dict)
            savename = f'{directory}/{subdirectory}/{filename[:-3]}png'
            local_dict['im'].save(savename)
            print(savename)


def divide_grid(filename, x_size, y_size):
    im = Image.open(filename)
    filename = os.path.splitext(filename)[0]

    directory = f'{filename}_divided{x_size:03d}_{y_size:03d}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_cols = im.size[0] // x_size
    num_rows = im.size[1] // y_size

    for i in range(num_rows):
        for j in range(num_cols):
            cropped = im.crop((j*x_size, i*y_size,
                               (j+1)*x_size, (i+1)*y_size))
            save_as = f'{directory}/{i:03d}_{j:03d}.png'

            print(save_as)
            cropped.save(save_as)


if __name__ == '__main__':
    # divide_grid(input(), int(input()), int(input()))

    for folder in os.listdir('A/'):
        filter_all(f'A/{folder}', 'create_val_to_color', ', -48, -108')


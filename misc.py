from PIL import Image, ImageFilter
import os


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


def filter_all(directory, subdirectory, additional_params=''):
    if not os.path.exists(f'{directory}/{subdirectory}'):
        os.makedirs(f'{directory}/{subdirectory}')

    print(subdirectory, directory)
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            print(filename)
            im = Image.open(directory + filename)
            local_dict = locals()
            exec(f'im = {subdirectory}(im{additional_params})', globals(), local_dict)
            local_dict['im'].save(f'{directory}/{subdirectory}/{filename[:-3]}png')


def divide_grid(filename, x_size, y_size):
    im = Image.open(filename)
    filename = os.path.splitext(filename)[0]

    dir = f'{filename}_divided{x_size:03d}_{y_size:03d}'
    if not os.path.exists(dir):
        os.makedirs(dir)

    num_cols = im.size[0] // x_size
    num_rows = im.size[1] // y_size

    for i in range(num_rows):
        for j in range(num_cols):
            cropped = im.crop((j*x_size, i*y_size,
                               (j+1)*x_size, (i+1)*y_size))
            save_as = f'{dir}/{i:03d}_{j:03d}.png'

            print(save_as)
            cropped.save(save_as)


if __name__ == '__main__':
    # divide_grid(input(), int(input()), int(input()))
    for folder in os.listdir('A/'):
        filter_all(f'A/{folder}/', 'unmoire_and_greyscale')


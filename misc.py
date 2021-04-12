from PIL import Image, ImageFilter
import os


def blur(directory, filename, amount):
    im = Image.open(directory + filename)
    im = im.convert('RGB')
    im = im.filter(ImageFilter.GaussianBlur(amount))
    im.save(f'{directory}/blur_{amount:d}/{filename[:-3]}png')


def blur_all(directory, amount):
    if not os.path.exists(f'{directory}/blur_{amount:d}'):
        os.makedirs(f'{directory}/blur_{amount:d}')

    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            blur(directory, filename, amount)


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
                               (j+1)*x_size-1, (i+1)*y_size-1))
            save_as = f'{dir}/{i:03d}_{j:03d}.png'
            print(save_as)
            cropped.save(save_as)


if __name__ == '__main__':
    # divide_grid(input(), int(input()), int(input()))
    divide_grid('large_portraits.png', 96, 80)

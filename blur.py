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


if __name__ == '__main__':
    blur_all('', 1)

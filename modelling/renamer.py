import os

def rename_images(images_dir, masks_dir):
    images = os.listdir(images_dir)
    masks = os.listdir(masks_dir)

    images.sort()
    masks.sort()

    for i in range(len(images)):
        image_ext = images[i].split('.')[-1]
        if image_ext == 'jpg':
            os.rename(os.path.join(images_dir, images[i]),
                      os.path.join(images_dir, '{:03d}.jpg'.format(i + 1)))

    for i in range(len(masks)):
        mask_ext = masks[i].split('.')[-1]
        if mask_ext == 'png':
            os.rename(os.path.join(masks_dir, masks[i]),
                      os.path.join(masks_dir, '{:03d}_label.PNG'.format(i + 1)))

#revise to data directory 
images_dir = './data/train_data/images'
masks_dir = './data/train_data/masks'
rename_images(images_dir, masks_dir)


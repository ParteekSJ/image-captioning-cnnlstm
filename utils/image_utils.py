from PIL import Image
import os


def reshape_image(image, shape):
    """Resize an image to the given shape."""
    return image.resize(shape, Image.ANTIALIAS)


def reshape_images(image_path, output_path, shape, logger):
    """Reshape the images in 'image_path' and save into 'output_path'."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images = os.listdir(image_path)
    num_im = len(images)
    for i, im in enumerate(images):
        with open(os.path.join(image_path, im), "r+b") as f:
            with Image.open(f) as image:
                image = reshape_image(image, shape)
                image.save(os.path.join(output_path, im), image.format)
        if (i + 1) % 100 == 0:
            logger.debug(
                "[{}/{}] Resized the images and saved into '{}'.".format(
                    i + 1, num_im, output_path
                )
            )


def reshape_all_images(config, logger):
    image_paths = [config.DATASET.IMAGE_DIR, config.DATASET.IMAGE_VAL_DIR]
    output_paths = [
        config.DATASET.RESIZED_IMAGE_DIR,
        config.DATASET.RESIZED_VAL_IMAGE_DIR,
    ]

    for image_path, output_path in zip(image_paths, output_paths):
        # Check if the output directory exists for the current set of paths
        if not os.path.exists(output_path):
            reshape_images(
                image_path=image_path,
                output_path=output_path,
                shape=[256, 256],
                logger=logger,
            )
            logger.info(
                f"All images in {image_path} resized and saved to {output_path}."
            )
        else:
            logger.info(
                f"Output directory {output_path} already exists. Skipping image reshaping for this set of images."
            )

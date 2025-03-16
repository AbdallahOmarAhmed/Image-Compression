from PIL import Image
import os
import matplotlib.pyplot as plt


def compress_image(image_path, output_format, quality):
    img = Image.open(image_path)
    compressed_path = f"compressed_{output_format.lower()}.{output_format.lower()}"
    img.save(compressed_path, format=output_format, quality=quality)
    return compressed_path


def visualize_images(original_path, compressed_paths):
    img_original = Image.open(original_path)
    images = [img_original] + [Image.open(path) for path in compressed_paths]
    titles = ["Original"] + [os.path.basename(path).split('.')[0].capitalize() for path in compressed_paths]

    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        img.show(title)
        # plt.subplot(1, len(images), i + 1)
        # plt.imshow(img)
        # plt.title(title)
        # plt.axis('off')
    # plt.show()


def main():
    image_path = "test/s5.jpeg"  # Replace with your image path
    formats = {"JPEG": 0, "WEBP": 0}  # HEIC unsupported by PIL

    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return

    original_size = os.path.getsize(image_path) / 1024
    print(f"Original size: {original_size:.2f} KB")

    compressed_paths = []
    for fmt, quality in formats.items():
        compressed_path = compress_image(image_path, fmt, quality)
        compressed_size = os.path.getsize(compressed_path) / 1024
        print(f"{fmt} compressed size: {compressed_size:.2f} KB")
        compressed_paths.append(compressed_path)

    visualize_images(image_path, compressed_paths)


if __name__ == "__main__":
    main()

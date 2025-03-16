from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt


def canny_edge_detection(image_path, low_threshold=50, high_threshold=150):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert("L")

    # Apply Gaussian blur to reduce noise
    blurred = img.filter(ImageFilter.GaussianBlur(radius=4))

    # Convert image to numpy array
    img_array = np.array(blurred)

    # Use numpy to apply a basic edge detection (manual implementation)
    from skimage import feature
    edges = feature.canny(img_array, sigma=1.0, low_threshold=low_threshold / 255.0,
                          high_threshold=high_threshold / 255.0)

    # Convert back to image for visualization
    edge_image = Image.fromarray((edges * 255).astype(np.uint8))
    return edge_image


# Example usage
if __name__ == "__main__":
    edge_img = canny_edge_detection("test/3.jpg", low_threshold=500, high_threshold=1500)
    plt.imshow(edge_img, cmap='gray')
    plt.title("Sensitive Canny Edge Detection")
    plt.axis('off')
    plt.show()
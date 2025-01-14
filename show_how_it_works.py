import numpy as np
import random
import cv2
import time
from region_growing import region_growing_step_by_step
from PIL import Image, ImageDraw
import ffmpeg

def generate_continuous_tentacles(
    random_state: int, 
    img_size: int = 512, 
    num_tentacles: int = 50
) -> Image:
    """
    Generates a grayscale image with smooth tentacles extending outward from the center.
    The image is generated at a higher resolution for anti-aliasing and then downsampled.

    Parameters:
        random_state (int): Seed for random number generation.
        img_size (int): Size of the output image (in pixels).
        num_tentacles (int): Number of tentacles to generate.

    Returns:
        Image: A PIL Image object representing the generated tentacles.
    """
    upscale_factor = 2  # Render at a higher resolution for smoothing
    working_size = img_size * upscale_factor

    # Initialize random seeds for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)

    # Create a blank grayscale image
    img = Image.new("L", (working_size, working_size), color=255)
    draw = ImageDraw.Draw(img)

    # Define the center of the image
    center = (working_size // 2, working_size // 2)

    # Parameters controlling tentacle appearance
    max_initial_thickness = int(0.06 * working_size)
    min_initial_thickness = int(0.02 * working_size)
    max_step_size = 0.012 * working_size
    min_step_size = 0.004 * working_size
    max_angle_variation = 0.15
    min_tentacle_length = int(0.15 * working_size)
    max_tentacle_length = int(0.40 * working_size)

    # Generate each tentacle
    for _ in range(num_tentacles):
        x, y = center  # Starting point at the center
        angle = random.uniform(0, 2 * np.pi)  # Initial random angle
        thickness = random.randint(min_initial_thickness, max_initial_thickness)
        tentacle_length = random.randint(min_tentacle_length, max_tentacle_length)

        # Draw tentacle segments
        for _ in range(tentacle_length):
            dir_x = np.cos(angle)  # Calculate direction
            dir_y = np.sin(angle)
            angle += random.uniform(-max_angle_variation, max_angle_variation)  # Randomize direction

            step_size = random.uniform(min_step_size, max_step_size)
            next_x = x + dir_x * step_size
            next_y = y + dir_y * step_size

            # Draw the segment
            draw.line([(x, y), (next_x, next_y)], fill=0, width=int(thickness))

            x, y = next_x, next_y  # Update position
            thickness -= random.uniform(0.2, 0.5)  # Gradually reduce thickness
            thickness = max(thickness, 1)  # Ensure minimum thickness

            # Stop if out of bounds
            if not (0 <= x < working_size and 0 <= y < working_size):
                break

    # Downsample the image to the target size with anti-aliasing
    resample_filter = Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS
    img = img.resize((img_size, img_size), resample=resample_filter)

    return img

def generate_video_with_region_growing(
    random_state: int,
    video_filename: str = 'tentacle_growth_video.mp4',
    img_size: int = 512,
    num_tentacles: int = 50,
    fps: int = 15,
    threshold: int = 20,
):
    """
    Creates a video of tentacle growth using a region-growing algorithm.

    Parameters:
        random_state (int): Seed for random number generation.
        video_filename (str): Output filename for the video.
        img_size (int): Image dimensions (width and height in pixels).
        num_tentacles (int): Number of tentacles in the generated image.
        fps (int): Frames per second for the video.
        threshold (int): Threshold value for region-growing segmentation.
    """
    start_time = time.time()

    # Generate the base tentacle image
    img = generate_continuous_tentacles(random_state, img_size, num_tentacles)
    img_np = np.array(img)  # Convert to a NumPy array
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (img_size, img_size))

    # Prepare the image for coloring regions (convert grayscale to RGB)
    img_rgb = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)

    # Initialize region-growing generator
    region_generator = region_growing_step_by_step(img_np, threshold)
    
    step_count = 0

    # Process and save each frame
    for region in region_generator:
        step_count += 1

        # Highlight current region in red
        img_rgb_copy = img_rgb.copy()
        img_rgb_copy[region] = [255, 0, 0]  # Set region to red

        # Write the frame to the video
        video_writer.write(img_rgb_copy)

    video_writer.release()  # Finalize the video

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Region growing completed after {step_count} steps.")
    print(f"Execution time: {execution_time:.2f} seconds.")

def convert_video_to_gif_ffmpeg(video_filename: str, gif_filename: str):
    """
    Converts a video to a GIF using ffmpeg.

    Parameters:
        video_filename (str): Path to the input video.
        gif_filename (str): Path to the output GIF.
    """
    (
        ffmpeg
        .input(video_filename)
        .output(gif_filename, vf='fps=15', loop=0)
        .run()
    )
    print(f"GIF saved as {gif_filename}")



if __name__ == "__main__":
    video_filename = "tentacle_growth_video.mp4"
    gif_filename = "tentacle_growth.gif"
    
    generate_video_with_region_growing(
        random_state=42,
        video_filename=video_filename,
        img_size=200,
        num_tentacles=50,
        fps=240,  # High FPS for smoother playback
        threshold=20
    )
    
    # Convert to GIF
    convert_video_to_gif_ffmpeg(video_filename, gif_filename)

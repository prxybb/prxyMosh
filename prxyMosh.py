import cv2
import numpy as np
import os

def process_video(input_path, division_size, budget, start_blank=False, fractal_level=1, refresh_rate=None):
    def update_squares(base_frame, curr_frame, squares, division_size):
        print(f"Updating {len(squares)} squares")
        for x, y in squares:
            print(f"Updating square at ({x}, {y})")
            base_frame[y:y+division_size, x:x+division_size] = curr_frame[y:y+division_size, x:x+division_size]
        return base_frame

    # load the video
    video = cv2.VideoCapture(input_path)

    # get video properties
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: Width = {width}, Height = {height}, FPS = {fps}")

    # check for unique output name
    output_path = 'output.mp4'
    counter = 1
    while os.path.isfile(output_path):
        output_path = f'output{counter}.mp4'
        counter += 1
    print(f"Output path: {output_path}")

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # read the first frame
    ret, prev_source_frame = video.read()
    if not ret:
        raise ValueError('Cannot read the first frame')

    if start_blank:
        base_frame = np.zeros_like(prev_source_frame)
        print("Start with blank frame")
    else:
        base_frame = prev_source_frame.copy()

    frame_count = 0
    while True:
        # read the current frame
        ret, curr_frame = video.read()
        if not ret:
            break  # end of the video

        # refresh background every X frames
        if refresh_rate is not None and frame_count % refresh_rate == 0:
            base_frame.fill(0)

        print(f"Processing frame #{frame_count}")
        frame_count += 1

        level_size = division_size * (2 ** fractal_level)
        selected_squares = [(x, y) for x in range(0, width, level_size) for y in range(0, height, level_size)]

        # fractal comparisons
        for level in range(fractal_level, 0, -1):
            print(f"Fractal level {level}, Level size {level_size}")
            diff_list = []
            for x, y in selected_squares:
                square_prev = prev_source_frame[y:y+level_size, x:x+level_size]
                square_curr = curr_frame[y:y+level_size, x:x+level_size]
                diff = np.sum(np.abs(square_curr.astype(int) - square_prev.astype(int)))
                diff_list.append((diff, x, y))

            diff_list.sort(reverse=True)
            print(f"Calculated differences for {len(diff_list)} squares")

            selected_squares = [(x+i, y+j) for _, x, y in diff_list[:int(len(diff_list) * (budget ** (1 / fractal_level)))] for i in range(0, level_size, level_size//2) for j in range(0, level_size, level_size//2) if x+i < width and y+j < height]
            print(f"Selected {len(selected_squares)} squares for level {level}")

            level_size //= 2  # reduce level size by 2 for next iteration

        # update the squares
        base_frame = update_squares(base_frame, curr_frame, selected_squares, division_size)

        # write the frame
        out.write(base_frame)

        # remember the current source frame for the next comparison
        prev_source_frame = curr_frame

    # release the video reader/writer
    video.release()
    out.release()

    # play the video
    os.system('open ' + output_path)

# usage
process_video('input.mov', 4, 0.15, True, 1, refresh_rate=1)

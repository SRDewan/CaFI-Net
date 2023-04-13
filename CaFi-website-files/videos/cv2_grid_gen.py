import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ExtractImageFromVideo(object):
    def __init__(self, path, frame_range=None, debug=False):
        assert os.path.exists(path)

        self._p = path
        self._vc = cv2.VideoCapture(self._p)

        self.size = int(self._vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self._vc.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self._vc.get(cv2.CAP_PROP_FRAME_COUNT))

        self._start = 0
        self._count = self.total_frames

        self._debug = debug

        if frame_range is not None:
            self.set_frames_range(frame_range)

        if self._debug:
            print(f"video size( W x H ) : {self.size[0]} x {self.size[1]}")

    def __del__(self):
        self.release()

    def set_frames_range(self, frame_range=None):
        if frame_range is None:
            self._start = 0
            self._count = self.total_frames
        else:
            assert isinstance(frame_range, (list, tuple, range))
            if isinstance(frame_range, (list, tuple)):
                assert len(frame_range) == 2

            start, end = frame_range[0], frame_range[-1]
            if end is None \
                    or end == -1 \
                    or end >= self.total_frames:
                end = self.total_frames
            assert end >= start

            self._start = start
            self._count = end - start
            assert self._count <= self.total_frames

        self._vc.set(cv2.CAP_PROP_POS_FRAMES, self._start)

    def extract(self, path=None, bgr2rgb=False, target_size=None, text=None, text_position=None):
        if path is not None and not os.path.exists(path):
            os.makedirs(path)

        for i in range(0, self._count):
            success, frame = self._vc.read()
            if not success:
                print(f"index {i} exceeded.")
                break
            if self._debug:
                print(f"frame {self._start + i}")
            if path is not None:
                cv2.imwrite(os.path.join(path, f"{self._start + i}.jpg"), frame)
            if bgr2rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if target_size is not None:
                assert len(target_size) == 2
                assert isinstance(target_size, (list, tuple))
                frame = cv2.resize(frame, tuple(target_size))
            if text is not None:
                if text_position is not None:
                    w_scale, h_scale = text_position
                else:
                    w_scale, h_scale = 1 / 10, 1 / 10
                pos = int(self.size[0] * w_scale), int(self.size[1] * h_scale)  # text position
                frame = cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=2)
            yield frame

    def release(self):
        if self._vc is not None:
            self._vc.release()

            
def create_image_grid(images, grid_size=None):
    border_size = 6
    split_border = 48
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-2], images.shape[-3]

    if grid_size is not None:
        grid_h, grid_w = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid_w += 1
    labels = ["bench", "cabinet", "car", "cellphone", "chair", "couch", "firearm", "lamp", "monitor", "plane", "speaker", "table", "watercraft"]

    grid = np.zeros([grid_h * img_h + (grid_h + 1) * border_size + img_h // 2, grid_w * img_w + (grid_w + 1) * border_size + split_border] + list(images.shape[-1:]), dtype=images.dtype)
    labels_width = img_w
    grid_width = grid.shape[1] - labels_width
    grid[:, : labels_width, ...] = np.array([255, 255, 255])
    grid[:, labels_width : labels_width + grid_width // 2, ...] = np.array([0, 62, 81])
    grid[:, labels_width + grid_width // 2 :, ...] = np.array([81, 62, 62])

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 3
    font_thickness = 3
    grid[:img_h // 2, :, ...] = np.array([255, 255, 255])

    (label_width, label_height), baseline = cv2.getTextSize("Input NeRFs", font, font_scale, font_thickness)
    inp_label_pos = (labels_width + (grid_width - split_border) // 4 - label_width // 2, img_h // 4 + label_height // 2)
    grid = cv2.putText(grid, "Input NeRFs", inp_label_pos, font, font_scale, (0, 0, 0), thickness = font_thickness)

    (label_width, label_height), baseline = cv2.getTextSize("Canonical Fields", font, font_scale, font_thickness)
    can_label_pos = (labels_width + (grid_width - split_border) // 2 + split_border + (grid_width - split_border) // 4 - label_width // 2, img_h // 4 + label_height // 2)
    grid = cv2.putText(grid, "Canonical Fields", can_label_pos, font, font_scale, (0, 0, 0), thickness = font_thickness)

    for idx in range(grid_w * grid_h):
        x = (idx % grid_w) * (img_w + border_size)
        y = (idx // grid_w) * (img_h + border_size) + border_size + img_h // 2

        if (idx % grid_w) == 0:
            label = labels[idx // grid_w]
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            font_pos = (x + img_w // 2 - label_width // 2, y + img_h // 2 + label_height // 2)
            grid = cv2.putText(grid, label, font_pos, font, font_scale, (0, 0, 0), thickness = font_thickness)
            continue

        # x -= img_w // 2
        if (idx % grid_w) >= 4:
            if (idx % grid_w) == 4:
                grid[:, x : x + split_border, ...] = np.array([255, 255, 255])

            x += split_border
            x += border_size

        img_idx = int(idx - np.ceil(idx / grid_w))
        grid[y : y + img_h, x : x + img_w, ...] = images[img_idx]

    # plt.imshow(grid)
    # plt.show()
    return grid


def merge_videos(videos_in, video_out, grid_size=None, titles=None, title_position=(0.5, 0.5), max_frames: int = None):
    """
    Args:
        videos_in: List/Tuple
            List of input video paths. e.g.
                ('path/to/v1.mp4', 'path/to/v2.mp4', 'path/to/v3.mp4')
        video_out: String
            Path of output video. e.g.
                'path/to/output.mp4'
        grid_size: List/Tuple.
            Row and Column respectively. e.g.
                (1, 3)
        titles: List/Tuple
            The title of each video will be displayed in the video grid,
            the same length as the input video. e.g.
                ('v1', 'v2', 'v3')
        title_position: List/Tuple
            The position(width and height) where the title is displayed, and the value range is (0, 1).
            e.g. If we want display text in the center of the video, the position is
                (0.5, 0.5)
        max_frames: Int
            Maximum number of frames per input video will be merge, e.g.
                200
    Returns:
        None
    """
    texts = titles
    if texts is None:
        texts = [None] * len(videos_in)
    assert len(videos_in) == len(texts)

    dir_name = os.path.dirname(video_out)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    video_handles = []
    for v, text in zip(videos_in, texts):
        assert os.path.exists(v), f'{v} not exists!'
        video_handles.append((ExtractImageFromVideo(v), text))

    if max_frames is not None:
        assert max_frames > 0
        least_frames = max_frames
    else:
        least_frames = sorted([e.total_frames for e, _t in video_handles])[0]  # all with same number of frames

    least_size = sorted([e.size for e, _t in video_handles])[0]  # all with same size WH
    least_size = (640, 360)
    generators = [e.extract(text=t, text_position=title_position) for e, t in video_handles]

    # read one frame and resize for each generator, then get the output video size
    cur_frames = np.array([cv2.resize(next(g), least_size) for g in generators])
    frames_grid = create_image_grid(cur_frames, grid_size=grid_size)  # HWC

    fps = video_handles[0][0].fps  # use the fps of first video
    out_size = frames_grid.shape[0:2]  # HWC to HW
    out_size = out_size[::-1]  # reverse HW to WH, as VideoWriter need that format
    video_writer = cv2.VideoWriter(video_out,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   fps,
                                   out_size)

    for n in range(least_frames-1):
        if n % 100 == 0:
            print(f'{n}: {len(cur_frames)} frames merge into grid with size={frames_grid.shape}')
        video_writer.write(frames_grid)

        cur_frames = np.array([cv2.resize(next(g), least_size) for g in generators])
        frames_grid = create_image_grid(cur_frames, grid_size=grid_size)

    video_writer.release()
    print(f'Output video saved... {video_out}')
    
list_videos = []
input_dir = "replacement_input_vids"
skip = []
for elem in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, elem)

    if not os.path.isdir(path) or elem in skip:
        continue

    list_videos.append([])
    for vid in sorted(os.listdir(path)):
        if ".mov" not in vid:
            continue

        vid_path = os.path.join(path, vid)
        list_videos[-1].append(vid_path)

input_dir = "canonical_vids"
ctr = 0
for elem in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, elem)

    if not os.path.isdir(path) or elem in skip:
        continue

    for vid in sorted(os.listdir(path)):
        if ".mov" not in vid or "_old" in vid:
            continue

        vid_path = os.path.join(path, vid)
        list_videos[ctr].append(vid_path)

    ctr += 1
videos_to_merge= [item for sublist in list_videos for item in sublist]

if __name__ == '__main__':
    # videos_to_merge = [
        # '1.mp4',
        # '2.mp4',
        # '3.mp4',
    # ]
    # titles = ['v1', 'v2', 'v3']
    titles = np.arange(len(videos_to_merge)).astype(str)
    merge_videos(
        videos_to_merge,
        'canonical_grid.mp4',
        grid_size = (13, 6),
        titles=None,
        title_position=(0.1, 0.5),  # text poistion is (0.1 * w, 0.5 * h)
        max_frames=100)  # merge first 100 frames per video

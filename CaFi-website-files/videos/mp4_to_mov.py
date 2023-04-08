import os

input_dir = "input_vids"
skip = ["old"]

for elem in os.listdir(input_dir):
    path = os.path.join(input_dir, elem)

    if not os.path.isdir(path) or elem in skip:
        continue

    for vid in os.listdir(path):
        vid_path = os.path.join(path, vid)
        out_path = vid_path[:-3] + "mov"
        cmd = "ffmpeg -i %s -f mov %s" % (vid_path, out_path)
        os.system(cmd)

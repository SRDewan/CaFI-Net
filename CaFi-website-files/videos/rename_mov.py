import os

input_dir = "input_vids"
skip = ["old"]

for elem in os.listdir(input_dir):
    path = os.path.join(input_dir, elem)

    if not os.path.isdir(path) or elem in skip:
        continue

    ctr = 1
    for vid in sorted(os.listdir(path)):
        if ".mov" not in vid:
            continue

        vid_path = os.path.join(path, vid)
        out_path = os.path.join(path, "input_%d.mov" % (ctr))
        cmd = "mv %s %s" % (vid_path, out_path)
        os.system(cmd)

        ctr += 1

import os

###list_videos contains the path to the videos
width = 1080
height = 1300
input_videos = ""
border_size = 4
input_setpts = "nullsrc=size={}x{} [base];".format(width + 5 * border_size, height + 12 * border_size)
input_overlays = "[base][video0] overlay=shortest=1 [tmp0];"
grid_width = 6
grid_height = 13

list_videos = []
input_dir = "input_vids"
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
        if ".mov" not in vid:
            continue

        vid_path = os.path.join(path, vid)
        list_videos[ctr].append(vid_path)

    ctr += 1

for i in range(len(list_videos)):
    curr_list_videos = list_videos[i]

    for ix, path_video in enumerate(curr_list_videos):
        index = 6 * i + ix
        input_videos += " -i " + path_video
        input_setpts += "[{}:v] setpts=PTS-STARTPTS, scale={}x{} [video{}];".format(index, width//grid_width, height//grid_height, index)
        if index > 0 and index < len(list_videos) * len(curr_list_videos) - 1:
            input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={} [tmp{}];".format(index-1, index, (width//grid_width + border_size) * (index%grid_width), (height//grid_height + border_size) * (index//grid_width), index)
        if index == len(list_videos) * len(curr_list_videos) - 1:
            input_overlays += "[tmp{}][video{}] overlay=shortest=1:x={}:y={}:fill=black".format(index-1, index, (width//grid_width + border_size) * (index%grid_width), (height//grid_height + border_size) * (index//grid_width))

complete_command = "ffmpeg" + input_videos + " -filter_complex \"" + input_setpts + input_overlays + "\" -c:v libx264 ffmpeg_merged.mp4"

print(complete_command) 
# os.system(complete_command)

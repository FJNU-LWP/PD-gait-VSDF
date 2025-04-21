import torch
import torch.nn as nn


def keypoint_patch_embed(x, coord):

    video_data = x

    coordinates = coord

    s = 32
    block_size = (3, s, s)

    batch_size, frame_count, _, height, width = video_data.size()

    result = torch.empty(batch_size, frame_count, 9, 3, s, s)

    for i in range(batch_size):
        for j in range(frame_count):
            for k in range(9):

                x, y = coordinates[i, j, k]


                left = int(x - s / 2)

                top = int(y - s / 2)

                left = max(0, min(left, 480))
                top = max(0, min(top, 480))


                block = video_data[i, j, :, top:top + s, left:left + s]
                '''
                block_np = block.permute(1, 2, 0).cpu().numpy()
                frame = block_np
                frame = (frame - frame.min()) / (frame.max() - frame.min())
                plt.imshow(frame)
                plt.show()
                #breakpoint()
                
                '''

                result[i, j, k] = block
    return result
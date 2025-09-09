blockDim_x = 32
blockDim_y = 32

for threadIdx_x in range(blockDim_x):
    for threadIdx_y in range(blockDim_y):
        in_block_id = threadIdx_x * blockDim_y + threadIdx_y
        reduce_gap = (blockDim_y - 1) // 2 + 1
        while reduce_gap > 0:
            if threadIdx_y < reduce_gap and threadIdx_y + reduce_gap < blockDim_y:
                print(f"thread x: {threadIdx_x}, thread y: {threadIdx_y}, block_id: {in_block_id}, reduce_gap: {in_block_id + reduce_gap}")
            if (reduce_gap == 1):
                break
            reduce_gap = (reduce_gap - 1) // 2 + 1

print("-" * 100)
for threadIdx_x in range(blockDim_x):
    for threadIdx_y in range(blockDim_y):
        in_block_id = threadIdx_x * blockDim_y + threadIdx_y
        reduce_gap = (blockDim_x - 1) // 2 + 1
        while reduce_gap > 0:
            if threadIdx_x < reduce_gap and threadIdx_y == 0 and threadIdx_x + reduce_gap < blockDim_x:
                print(f"thread x: {threadIdx_x}, thread y: {threadIdx_y}, block_id: {in_block_id}, reduce_gap: {in_block_id + reduce_gap * blockDim_y}")
            if (reduce_gap == 1):
                break
            reduce_gap = (reduce_gap - 1) // 2 + 1

#         // Reduction on the y dimension.
#         int in_block_id = threadIdx.x * blockDim.y + threadIdx.y;
#         int reduce_gap = (blockDim.y - 1) / 2 + 1;
#         while (reduce_gap > 0) {
#             if (threadIdx.y < reduce_gap && threadIdx.y + reduce_gap < blockDim.y) {
#                 max_sh[in_block_id] = fmaxf(max_sh[in_block_id], max_sh[in_block_id + reduce_gap]);
#                 min_sh[in_block_id] = fminf(min_sh[in_block_id], min_sh[in_block_id + reduce_gap]);
#             }
#             __syncthreads();
#             if (reduce_gap == 1) {
#                 break;
#             }
#             reduce_gap = (reduce_gap - 1) / 2 + 1;
#         }
#         // Reduction on the x dimension.
#         reduce_gap = (blockDim.x - 1) / 2 + 1;
#         while (reduce_gap > 0) {
#             if (threadIdx.x < reduce_gap && threadIdx.y == 0 && threadIdx.x + reduce_gap < blockDim.x) {
#                 max_sh[in_block_id] = fmaxf(max_sh[in_block_id], max_sh[in_block_id + reduce_gap * blockDim.y]);
#                 min_sh[in_block_id] = fminf(min_sh[in_block_id], min_sh[in_block_id + reduce_gap * blockDim.y]);
#             }
#             __syncthreads();
#             if (reduce_gap == 1) {
#                 break;
#             }
#             reduce_gap = (reduce_gap - 1) / 2 + 1;
#         }

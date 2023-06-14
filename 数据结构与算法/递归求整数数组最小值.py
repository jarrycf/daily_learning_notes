def find_min(arr, start, end):
    if start == end:  # 如果区间中只有一个数，返回该数
        return arr[start]
    mid = (start + end) // 2  # 将区间平均分成两半，找到中间位置mid
    left_min = find_min(arr, start, mid)  # 在左半部分递归查找最小值
    right_min = find_min(arr, mid + 1, end)  # 在右半部分递归查找最小值
    return min(left_min, right_min)  # 返回左右两半的最小值
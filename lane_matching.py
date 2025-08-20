import numpy as np

LANE_MAX_DIST = 1280

def get_lane_distance(lane_a_pts, lane_b_pts):
    if len(lane_a_pts) == 0 or len(lane_b_pts) == 0:
        return LANE_MAX_DIST

    lane_a_start_v = lane_a_pts[0][1]
    lane_a_end_v = lane_a_pts[-1][1]
    lane_b_start_v = lane_b_pts[0][1]
    lane_b_end_v = lane_b_pts[-1][1]

    overlap_start_v = max(lane_a_start_v, lane_b_start_v)
    overlap_end_v = min(lane_a_end_v, lane_b_end_v)
    overlap_length = overlap_end_v - overlap_start_v

    if overlap_length <= 0:
        return LANE_MAX_DIST

    unoverlap_start_v = min(lane_a_start_v, lane_b_start_v)
    unoverlap_end_v = max(lane_a_end_v, lane_b_end_v)
    unoverlap_length = unoverlap_end_v - unoverlap_start_v
    if overlap_length / unoverlap_length < 0.25:
        return LANE_MAX_DIST

    # 计算overlap_start_v到overlap_end_v之间, lane_a和lane_b的距离平均值
    #构建一个720长度的一维数组，初始值为0
    new_lane_a_pts = np.zeros(720)
    new_lane_b_pts = np.zeros(720)
    for u, v in lane_a_pts:
        new_lane_a_pts[v] = u
    for u, v in lane_b_pts:
        new_lane_b_pts[v] = u
    
    # 计算overlap_start_v到overlap_end_v之间, new_lane_a_pts和new_lane_b_pts的距离平均值
    distance_avg = 0
    for v in range(overlap_start_v, overlap_end_v, 10):
        distance = np.abs(new_lane_a_pts[v] - new_lane_b_pts[v])
        distance_avg += distance
    distance_avg /= (overlap_end_v - overlap_start_v) / 10

    return distance_avg


def get_lane_dist2center(lane_pts):
    if len(lane_pts) == 0:
        return LANE_MAX_DIST
    center_u = 1280 / 2
    dist_sum = 0
    for u, v in lane_pts:
        dist = abs(u - center_u)
        dist_sum += dist
    dist_avg = dist_sum / len(lane_pts)
    return dist_avg


def remove_close_lanes(lanes_points, dist_thr=80, verbose=False):
    # 1. 计算lane的长度
    lane_lengths = []
    for lane in lanes_points:
        if len(lane) < 2:
            lane_lengths.append(0)
            continue
        first_point = lane[0]
        last_point = lane[-1]
        lane_length = last_point[1] - first_point[1]
        lane_lengths.append(lane_length)

    if verbose:
        print(f"lane_lengths: {lane_lengths}")

    lane_dist2center = []
    for lane in lanes_points:
        dist = get_lane_dist2center(lane)
        lane_dist2center.append(dist)

    if verbose:
        print(f"lane_dist2center: {lane_dist2center}")

    # 2. 计算lane之间的距离
    lane_distances = [[0 for _ in range(len(lane_lengths))] for _ in range(len(lane_lengths))]
    for i in range(len(lane_lengths)):
        for j in range(i+1, len(lane_lengths)):
            distance = get_lane_distance(lanes_points[i], lanes_points[j])
            if verbose:
                print(f"lane {i} and lane {j} distance: {distance}")
            lane_distances[i][j] = distance
            lane_distances[j][i] = distance
    
    # 4. 如果lane之间的距离小于dist_thr，则认为lane是匹配的，只保留长度较长的lane
    lane_idx_list = range(len(lane_lengths))
    lane_idx_set = set(lane_idx_list)
    for i in range(len(lane_lengths)):
        for j in range(i+1, len(lane_lengths)):
            if lane_distances[i][j] < dist_thr:
                if verbose:
                    print(f"lane {i} and lane {j} are close, distance: {lane_distances[i][j]}")
                delta_length = abs(lane_lengths[i] - lane_lengths[j])
                if delta_length > 40:
                    # 如果lane的长度差大于40，则保留长度较长的lane
                    if lane_lengths[i] > lane_lengths[j]:
                        lane_idx_set.remove(j)
                        if verbose:
                            print(f"remove lane {j}")
                    else:
                        lane_idx_set.remove(i)
                        if verbose:
                            print(f"remove lane {i}")
                else:
                    # 保留靠近图像中间的lane
                    if lane_dist2center[i] < lane_dist2center[j]:
                        lane_idx_set.remove(j)
                        if verbose:
                            print(f"remove lane {j}")
                    else:
                        lane_idx_set.remove(i)
                        if verbose:
                            print(f"remove lane {i}")

    if verbose:
        print(f"lane_idx_set left: {lane_idx_set}")
    new_lanes_points = [lanes_points[i] for i in lane_idx_set]
    return new_lanes_points


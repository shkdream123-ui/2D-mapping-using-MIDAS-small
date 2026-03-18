# Python 3.12
import socket
import struct
import numpy as np
import cv2
from collections import deque
import threading
import math
import time
import scipy
from skimage.morphology import skeletonize

HOST = '0.0.0.0'
PORT = 5000

EXPECTED_W = 256
EXPECTED_H = 256

raw_queue = deque(maxlen=1)
processed_queue = deque(maxlen=1)
motion_queue = deque(maxlen=1)
global_queue = deque(maxlen=1)
gyro_queue  = deque(maxlen=50)
accel_queue  = deque(maxlen=50)
delta_yaw_queue = deque(maxlen=10)
corrected_queue = deque(maxlen=1)
pulse_queue = deque(maxlen=10)

# ------------------- TCP 수신 -------------------
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_thread(sock, raw_queue, gyro_queue, accel_queue):
    """
    Packet format
    ------------------------------------------------
    [1B] packet_type
      0x01 : frame packet
        [4B] jpeg_len
        [4B] depth_len
        [jpeg bytes]
        [depth bytes]

      0x02 : gyro packet
        [8B] timestamp (ns)
        [4B] gyro_z (rad/s)

      0x03 : accel packet
        [8B] timestamp (ns)
        [4B] accel_x (m/s^2)
        [4B] accel_y (m/s^2)
        [4B] accel_z (m/s^2)

    ------------------------------------------------
    """

    while True:
        # -----------------------------
        # 1) packet type
        # -----------------------------
        pkt_type_raw = recv_all(sock, 1)
        if not pkt_type_raw:
            print("[RECV] connection closed")
            break

        pkt_type = pkt_type_raw[0]

        # -----------------------------
        # 2) FRAME PACKET
        # -----------------------------
        if pkt_type == 0x01:
            header = recv_all(sock, 8)
            if not header:
                break

            jpeg_len, depth_len = struct.unpack('!II', header)

            jpeg_bytes = recv_all(sock, jpeg_len)
            depth_bytes = recv_all(sock, depth_len)

            if jpeg_bytes is None or depth_bytes is None:
                print("[RECV] frame incomplete")
                break

            raw_queue.append(
                (jpeg_bytes, depth_bytes)
            )

        # -----------------------------
        # 3) GYRO PACKET
        # -----------------------------
        elif pkt_type == 0x02:
            payload = recv_all(sock, 12)
            if not payload:
                break

            timestamp, gyro_z = struct.unpack('!qf', payload)

            #print(f"Received gyro: timestamp={timestamp}, gyroZ={gyro_z}")

            gyro_queue.append(
                (timestamp, gyro_z)
            )

        # ------------------------------
        # 4) ACCEL PACKET
        # ------------------------------
        elif pkt_type == 0x03:
            payload = recv_all(sock, 20)
            if not payload:
                break

            timestamp, ax, ay, az = struct.unpack('!qfff', payload)

            pc_ts = time.monotonic_ns()

            #print(f"Received accel: timestamp={timestamp}, ax={ax}, ay={ay}, az={az}")

            accel_queue.append(
                (timestamp, pc_ts, ax, ay, az)
            )


        else:
            print(f"[RECV] unknown packet type: {pkt_type}")
            break

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


# --------------------1d lidar 공간적 신뢰도 계산 함수 ----------------------------
def compute_spatial_confidence(lidar_1d):
    n = len(lidar_1d)
    conf = np.zeros(n, dtype=np.float32)

    valid = lidar_1d > 0
    if np.count_nonzero(valid) < 3:
        return conf

    # continuity
    cont = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        if valid[i-1] and valid[i] and valid[i+1]:
            cont[i] = abs(lidar_1d[i] - lidar_1d[i-1]) + \
                      abs(lidar_1d[i] - lidar_1d[i+1])

    sigma_cont = np.percentile(cont[cont > 0], 80) + 1e-6
    conf_cont = np.exp(-cont / sigma_cont)

    # slope smoothness
    grad = np.diff(lidar_1d)
    curv = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        if valid[i-1] and valid[i] and valid[i+1]:
            curv[i] = abs(grad[i] - grad[i-1])

    sigma_slope = np.percentile(curv[curv > 0], 80) + 1e-6
    conf_slope = np.exp(-curv / sigma_slope)

    # 🔥 핵심 변경
    conf = 0.6 * conf_cont + 0.4 * conf_slope
    conf[~valid] = 0.0
    conf[0] = conf[-1] = 0.2

    return np.clip(conf, 0.0, 1.0)

# -------------------------------------------------------------------------
def compute_spatial_hist_sim(mask_curr, mask_prev, G=3, eps=1e-6):
    """
    mask_curr, mask_prev : boolean mask (H, W)
    return : cosine similarity of spatial histograms
    """
    H, W = mask_curr.shape

    hist_curr = np.zeros((G, G), dtype=np.float32)
    hist_prev = np.zeros((G, G), dtype=np.float32)

    ys, xs = np.where(mask_curr)
    ys_p, xs_p = np.where(mask_prev)

    if len(xs) == 0 or len(xs_p) == 0:
        return 0.0

    for y, x in zip(ys, xs):
        hy = min(G-1, int(y * G / H))
        hx = min(G-1, int(x * G / W))
        hist_curr[hy, hx] += 1.0

    for y, x in zip(ys_p, xs_p):
        hy = min(G-1, int(y * G / H))
        hx = min(G-1, int(x * G / W))
        hist_prev[hy, hx] += 1.0

    hist_curr /= hist_curr.sum()
    hist_prev /= hist_prev.sum()

    h1 = hist_curr.flatten()
    h2 = hist_prev.flatten()

    return np.dot(h1, h2) / (
        np.linalg.norm(h1) * np.linalg.norm(h2) + eps
    )
#-------------------------------------------------------------------------
def cosine_similarity(a, b, eps=1e-8):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps)

#------------------------------------------------------------------------------------

def estimate_scale_orb(
    gray_prev, gray_curr,
    depth_prev, depth_curr,
    valid_mask_curr,
    trans_amount,
    min_matches=20
):
    orb = cv2.ORB_create(
        nfeatures=800,
        scaleFactor=1.2,
        nlevels=4
    )

    # -------------------------
    # mask 처리 (None 허용)
    # -------------------------
    if valid_mask_curr is None:
        mask = None
    else:
        mask = cv2.resize(
            valid_mask_curr.astype(np.uint8) * 255,
            gray_curr.shape[::-1],
            interpolation=cv2.INTER_NEAREST
        )

    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, mask)

    if des1 is None or des2 is None:
        return None


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < min_matches:
        return None

    disps = []
    for m in matches:
        p1 = np.array(kp1[m.queryIdx].pt)
        p2 = np.array(kp2[m.trainIdx].pt)

        d = np.linalg.norm(p2 - p1)
        if 1.0 < d < 80.0:
            disps.append(d)

    if len(disps) < min_matches:
        return None

    pixel_disp = np.median(disps)

    # 상대 스케일 (trans_amount는 기준만 제공)
    scale_orb = trans_amount / (pixel_disp + 1e-6)
    return scale_orb
#---------------------------------------------------------------------------------------

def resize_depth_to_frame(depth_uint8, gray_shape):
    """
    depth_uint8 : (Hd, Wd)  — MiDaS depth (e.g. 64x64)
    gray_shape  : (H, W)    — gray frame shape

    return      : (H, W) resized depth (float32)
    """
    H, W = gray_shape[:2]

    depth_resized = cv2.resize(
        depth_uint8.astype(np.float32),
        (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    return depth_resized
#------------------------------------------------------------------------------------
def compute_absolute_scale_k(
    trans_amount,
    pixel_translation,
    inlier_disp,      # (N,) pixel displacement
    inlier_pts,       # (N, 2) pixel coords (optional)
    depth_uint8,      # MiDaS depth map (uint8 or float)
    gray_shape
):
    import numpy as np

    # -----------------------------
    # 1️⃣ Motion-based depth (sparse)
    # -----------------------------
    disp = np.asarray(inlier_disp)

    # 너무 작은 시차 제거
    valid = disp > 1.0
    if np.count_nonzero(valid) < 10:
        return None

    disp = disp[valid]

    # Z = T / disparity
    Z_motion = trans_amount / disp   # [m]

    # robust statistic
    Z_motion_med = np.median(Z_motion)

    # -----------------------------
    # 2️⃣ MiDaS depth median
    # -----------------------------
    depth = depth_uint8.astype(np.float32)

    # 0 / invalid 제거
    depth_valid = depth[depth > 1e-3]
    if depth_valid.size < 100:
        return None

    Z_midas_med = np.median(depth_valid)

    # -----------------------------
    # 3️⃣ k_abs
    # -----------------------------
    raw_k = Z_motion_med / Z_midas_med

    GAIN = 30.0   # 실험 기반 (지금 관측치에 정확히 대응)
    k_abs = raw_k * GAIN

    # sanity check
    if not np.isfinite(k_abs) or k_abs < 1e-6 or k_abs > 1e3:
        return None

    return k_abs

#------------------------------------------------------------------------------------
def estimate_pixel_translation_and_inliers(
    gray_prev, gray_curr,
    max_features=1000,
    min_inliers=20,
    ransac_thresh=2.0,
    STATIC_DISP_THR=0.5   # ⭐ 정지 임계값 (pixel)
):
    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    # 특징점 부족 → 정지로 간주
    if des1 is None or des2 is None:
        return 0.0, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 매칭 부족 → 정지
    if len(matches) < min_inliers:
        return 0.0, None

    displacements = []
    pts_curr = []

    for m in matches:
        p1 = np.array(kp1[m.queryIdx].pt)
        p2 = np.array(kp2[m.trainIdx].pt)
        d = p2 - p1
        displacements.append(d)
        pts_curr.append(p2)

    displacements = np.array(displacements)
    pts_curr = np.array(pts_curr)

    # -------------------------
    # RANSAC
    # -------------------------
    best_inliers = None
    best_count = 0

    for _ in range(50):  # 100 → 50도 충분
        idx = np.random.randint(len(displacements))
        model = displacements[idx]
        errors = np.linalg.norm(displacements - model, axis=1)
        inliers = errors < ransac_thresh

        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_inliers = inliers

    # RANSAC 실패 → 정지
    if best_inliers is None or best_count < min_inliers:
        return 0.0, None

    inlier_disp = displacements[best_inliers]
    inlier_pts = pts_curr[best_inliers]

    pixel_translation = np.median(
        np.linalg.norm(inlier_disp, axis=1)
    )

    # ⭐ 최종 정지 판정
    if pixel_translation < STATIC_DISP_THR:
        return 0.0, None

    return pixel_translation, (inlier_disp, inlier_pts)


# ------------------- Depth 스레드 (최신 frame + 2D 변환 포함) -------------------


def depth_thread():
    import time
    import cv2
    import numpy as np

    prev_depth = None
    prev_depth_uint8 = None
    prev_scale = None
    valid_mask_prev = None

    depth_alpha = 0.6
    smoothing_skip = 2
    frame_counter = 0

    eps = 1e-6

    V_THR = 0.5
    DEPTH_SIM_THR = 0.7
    FRAME_SIM_THR = 0.6
    diff_thr = 15
    MIN_RATIO_PIXELS = 80

    MIN_DEPTH = 5
    MAX_DEPTH = 250

    depth_valid_prev = False
    valid_reason = None

    var_x = None
    var_y = None
    spatial_var = None
    theta_var = None
    mean_grad = None
    valid_reason = "UNDEFINED"
    k_abs = 0
    skel_ratio = None


    # -------------------------
    # Thresholds (초기값)
    # -------------------------
    MIN_EDGE_PTS = 30

    MIN_SPATIAL_VAR = 200.0        # edge가 충분히 퍼져 있는가
    ANISO_THR = 2.0                # var_x / var_y 비대칭성
    THETA_VAR_THR = 0.5            # 방향 분산 (rad^2)
    
    GRAD_ABS_MIN = 4.0

    MIN_EDGE_PIXELS = 120
    MIN_SKEL_PIXELS = 10
    SKEL_RATIO_MIN = 0.08
    SKEL_RATIO_MAX = 0.3
    MAX_SKEL_COMPONENTS = 4
    MIN_SKEL_SPATIAL_VAR = 30

    DEPTH_SIM_THR = 0.6
    MAD_K = 2.5
    MAD_MAX = 0.15

    absolute_scale_fixed = False
    pulse_active = False

    pixel_translation_acc = 0.0
    prev_pixel_translation = None

    frame_idx = 0

    SYNC_WINDOW_NS = 1_000_000_000_000

    debug = False
    visual = False
    

    def clamp01(x):
        return max(0.0, min(1.0, float(x)))

    while True:

        try:
            img, depth, yaw_acc, state, S_static, energy, direction, current_pc_ts = motion_queue.popleft()
        except IndexError:
            continue

        if img is None or img.size == 0:
            print("[depth_thread] empty img, skip")
            continue

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            print("[depth_thread] invalid img, skip")
            continue

        if depth is None or not isinstance(depth, np.ndarray):
            print("[depth_thread] invalid depth, skip")
            continue

        now_ts = time.monotonic_ns()
        PULSE_TIME_WINDOW = 0.15  # 150ms

        frame_pc_ts = now_ts
        
        if current_pc_ts is not None:
            pulse_active = (
                abs(now_ts - current_pc_ts) < PULSE_TIME_WINDOW
            )

        #print(trans_amount)

        depth_array = np.frombuffer(depth, dtype='<f4').copy()
        if depth_array.size != EXPECTED_W * EXPECTED_H:
            continue

        depth_frame = depth_array.reshape((EXPECTED_H, EXPECTED_W))
        depth_small = cv2.resize(depth_frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_counter += 1

        # =========================
        # Temporal smoothing
        # =========================
        if prev_depth is None:
            smoothed = depth_small.copy()
        else:
            if frame_counter % smoothing_skip == 0:
                smoothed = cv2.addWeighted(
                    prev_depth.astype(np.float32), depth_alpha,
                    depth_small.astype(np.float32), 1 - depth_alpha, 0.0
                )
            else:
                smoothed = depth_small.copy()

        prev_depth = smoothed.copy()

        # =========================
        # Normalize → uint8 (relative depth)
        # =========================
        min_val = np.min(smoothed)
        max_val = np.max(smoothed)
        range_val = max(max_val - min_val, 1e-6)

        depth_uint8 = ((smoothed - min_val) / range_val * 255).astype(np.uint8)

        # =========================
        # 2D → 1D LiDAR projection
        # =========================
        h, w = depth_uint8.shape
        y_positions = np.full(w, -1, dtype=np.int32)

        for x in range(w):
            column = depth_uint8[:, x]
            if column.max() > 0:
                y = np.argmax(column)
                y_positions[x] = (h - 1) - y

        # =========================
        # 1D smoothing
        # =========================
        valid_idx = (y_positions >= 0)
        if np.any(valid_idx):
            y_filled = y_positions.copy()
            last = y_filled[valid_idx][0]

            for i in range(w):
                if y_filled[i] == -1:
                    y_filled[i] = last
                else:
                    last = y_filled[i]

            y_smooth = cv2.GaussianBlur(
                y_filled.reshape(1, -1).astype(np.float32),
                ksize=(1, 9),
                sigmaX=1.8
            ).flatten().astype(np.int32)
        else:
            y_smooth = y_positions

        # =========================
        # lidar_2d (relative depth)
        # =========================
        lidar_2d = np.zeros(w, dtype=np.float32)
        for x in range(w):
            ys = y_smooth[x]
            if 0 <= ys < h:
                lidar_2d[x] = float(depth_small[ys, x])

        # =========================
        # spatial confidence (이미 존재)
        # =========================
        lidar_conf = compute_spatial_confidence(lidar_2d)

        # =========================
        # ⭐ depth validity(scale gate) ⭐
        # =========================
        # --- 1. Sobel gradient ---
        gx = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)

        theta = np.arctan2(gy, gx)

        # --- 2. 통계량 ---
        grad_flat = grad_mag.flatten()
        grad_flat = grad_flat[~np.isnan(grad_flat)]

        if grad_flat.size < 2:
            depth_valid_curr = False
            valid_reason = "INSUFFICIENT_GRAD_PIXELS"

            mean = std = var = 0.0
        else:
            mean = np.mean(grad_flat)
            std  = np.std(grad_flat)
            var  = np.var(grad_flat)

        if not np.isfinite(grad_mag).any():
            depth_valid_curr = False
            valid_reason = "GRAD_ALL_NAN"

        p25, p50, p75 = np.percentile(grad_flat, [25, 50, 75])

        T = np.percentile(grad_mag, 90)

        depth_valid_curr = False

        if T > GRAD_ABS_MIN: #상위 10퍼센트의 gradient가 유효할 때  
        
            if grad_mag is not None:
                
                edge_mask = grad_mag > T

                margin = int(min(h, w) * 0.2)

                edge_mask[:margin, :] = 0
                edge_mask[-margin:, :] = 0
                edge_mask[:, :margin] = 0
                edge_mask[:, -margin:] = 0

                edge_vals = grad_mag[edge_mask]

                theta_edges = theta[edge_mask]
                
                if theta_edges.size >= 2:
                    theta_var = np.var(theta_edges)
                else:
                    theta_var = np.inf   # 또는 충분히 큰 값
                    
                if edge_vals.size >= 2:
                    mean_grad = grad_mag[edge_mask].mean()
                else:
                    mean_grad = 0.0

                skel = skeletonize(edge_mask).astype(np.uint8)

                edge_count = edge_mask.sum()
                skel_count = skel.sum()

                ys, xs = np.where(edge_mask)
                
                if len(xs) >10 and len(ys) > 10:
                    var_x = np.var(xs)
                    var_y = np.var(ys)
                    spatial_var = var_x + var_y
                    anisotropy = max(var_x, var_y) / (min(var_x, var_y) + eps)

                    # =========================
                    # 1️⃣ Primary gate: 윤곽 기반
                    # =========================
                    if edge_count > MIN_EDGE_PIXELS and skel_count > MIN_SKEL_PIXELS:

                        skel_ratio = skel_count / (edge_count + eps)

                        num_labels, _ = cv2.connectedComponents(skel)
                        num_components = num_labels - 1

                        ys_s, xs_s = np.where(skel)
                        if len(xs_s) >= 2:
                            skel_spatial_var = np.var(xs_s) + np.var(ys_s)
                        else:
                            skel_spatial_var = 0.0

                        if (
                            SKEL_RATIO_MIN < skel_ratio < SKEL_RATIO_MAX and
                            num_components <= MAX_SKEL_COMPONENTS and
                            skel_spatial_var > MIN_SKEL_SPATIAL_VAR
                        ):
                            depth_valid_curr = True
                            valid_reason = "STRUCTURAL_SKELETON"

                    # =========================
                    # 2️⃣ Secondary gate: 방향성
                    # =========================
                    else:
                        if len(theta_edges) > MIN_EDGE_PTS and np.isfinite(theta_var):

                            if theta_var < THETA_VAR_THR:#방향이 일관적인 경우
                                depth_valid_curr = True
                                valid_reason = "DIRECTIONAL_SLOPE"

        # -------------------------
        # 결과 및 뎁스 유사여부
        # -------------------------
        if not depth_valid_curr:
            valid_reason = "INVALID_DEPTH"
            sim = None

        else:
            # 현재 depth에서 histogram은 항상 계산
            hist_curr, _ = np.histogram(
                theta_edges, bins=8, range=(-np.pi, np.pi), density=True
            )

            # 이전 depth도 유효할 때만 비교
            if depth_valid_prev is True:
                prev_gx = cv2.Sobel(prev_depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
                prev_gy = cv2.Sobel(prev_depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
                prev_grad_mag = np.sqrt(prev_gx**2 + prev_gy**2)

                prev_theta = np.arctan2(prev_gy, prev_gx)

                prev_T = np.percentile(prev_grad_mag, 90)

                prev_edge_mask = prev_grad_mag > prev_T

                prev_edge_mask[:margin, :] = 0
                prev_edge_mask[-margin:, :] = 0
                prev_edge_mask[:, :margin] = 0
                prev_edge_mask[:, -margin:] = 0

                theta_edges_prev = prev_theta[prev_edge_mask]
                
                hist_prev, _ = np.histogram(
                    theta_edges_prev, bins=8, range=(-np.pi, np.pi), density=True
                )
                
                hc = hist_curr.astype(np.float32)
                hp = hist_prev.astype(np.float32)

                hc /= (hc.sum() + 1e-8)
                hp /= (hp.sum() + 1e-8)

                sim = cosine_similarity(hc, hp)
            else:
                sim = None  # 비교 불가


        # --- 시각화용 정규화 ---
        grad_vis = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX)
        grad_vis = grad_vis.astype(np.uint8)
        grad_vis = cv2.applyColorMap(grad_vis, cv2.COLORMAP_INFERNO)
        depth_uint8_resized = cv2.resize(depth_uint8, (256, 256), interpolation=cv2.INTER_AREA)
        grad_vis_resized = cv2.resize(grad_vis, (256, 256), interpolation=cv2.INTER_AREA)

        # --- 출력 ---
        if visual is True:
            cv2.imshow("depth", depth_uint8_resized)
            cv2.imshow("gradient", grad_vis_resized)
            cv2.waitKey(1)

        scale = 1.0

        # 1️⃣ bytes → numpy buffer
        #np_buf = np.frombuffer(img, dtype=np.uint8)

        # 2️⃣ JPEG decode
        #frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

        # 3️⃣ BGR → Gray
        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        depth_curr = cv2.normalize(
            depth_uint8, None,
            alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX
        ).astype(np.uint8)


        if prev_scale is not None:

            if (
                valid_reason == "STRUCTURAL_SKELETON" and
                prev_valid_reason == "STRUCTURAL_SKELETON" and
                sim is not None and
                sim >= DEPTH_SIM_THR
            ):

                # =========================
                # 1️⃣ 후보 overlap mask 생성
                # =========================
                # 윤곽이 있는 곳 중 어느 한쪽이라도 구조가 있는 영역
                candidate_mask = edge_mask | prev_edge_mask

                # depth 유효성 (픽셀 단위)
                depth_valid_curr_px = (
                    (depth_curr > MIN_DEPTH) &
                    (depth_curr < MAX_DEPTH) &
                    (~np.isnan(depth_curr))
                )

                depth_valid_prev_px = (
                    (depth_prev > MIN_DEPTH) &
                    (depth_prev < MAX_DEPTH) &
                    (~np.isnan(depth_prev))
                )

                candidate_mask &= depth_valid_curr_px & depth_valid_prev_px

                if candidate_mask.sum() < MIN_RATIO_PIXELS:
                    scale = prev_scale

                else:
                    # =========================
                    # 2️⃣ depth ratio 계산
                    # =========================
                    d_curr = depth_curr[candidate_mask]
                    d_prev = depth_prev[candidate_mask]

                    ratio = d_curr / (d_prev + eps)

                    # 1차 물리적 범위 필터
                    ratio = ratio[(ratio > 0.3) & (ratio < 3.0)]

                    if ratio.size < MIN_RATIO_PIXELS:
                        scale = prev_scale

                    else:
                        # =========================
                        # 3️⃣ 강력한 통계 필터 (MAD)
                        # =========================
                        median = np.median(ratio)
                        mad = np.median(np.abs(ratio - median)) + eps

                        inlier = np.abs(ratio - median) < MAD_K * mad
                        ratio_filt = ratio[inlier]

                        # =========================
                        # 4️⃣ 품질 검사
                        # =========================
                        if (
                            ratio_filt.size < MIN_RATIO_PIXELS or
                            mad > MAD_MAX
                        ):
                            scale = prev_scale
                        else:
                            scale = median

            elif (
                valid_reason == "STRUCTURAL_SKELETON" and
                prev_valid_reason == "STRUCTURAL_SKELETON" and
                sim is not None and
                sim < DEPTH_SIM_THR
            ):
                scale_orb = estimate_scale_orb(
                        gray_prev, gray_curr,
                        prev_depth_uint8, depth_uint8,
                        None,
                        trans_amount
                    )

                if scale_orb is not None:
                    scale = 0.7 * prev_scale + 0.3 * scale_orb
                else:
                    scale = prev_scale

            else:
                scale = prev_scale

        # ----- 스파이크를 통한 국소적 동기화 -----

        if prev_scale is not None:
            pixel_translation, inliers = estimate_pixel_translation_and_inliers(
                gray_prev, gray_curr
            )
        else:
            pixel_translation = prev_pixel_translation

        # delta 계산
        if prev_pixel_translation is None:
            delta_pixel_translation = 0.0
        else:
            delta_pixel_translation = pixel_translation - prev_pixel_translation

        #print(delta_pixel_translation)

        # -------------------------------------------------
        # Spike detector (depth_thread local state)
        # -------------------------------------------------

        # threshold 설정 (튜닝 대상)
        NOISE_THR = 3.0
        PEAK_THR  = 10.0
        MAX_SPIKE_LEN = 10   # 프레임 수 (아주 짧게)

        # -------------------------
        # 상태 초기화 (최초 1회)
        # -------------------------
        if not hasattr(depth_thread, "spike_state"):
            depth_thread.spike_state = "IDLE"
            depth_thread.spike_pos_peak = 0.0
            depth_thread.spike_neg_peak = 0.0
            depth_thread.spike_len = 0

            depth_thread.spike_buffer = []          # [(frame_idx, gray, depth)]
            depth_thread.spike_pos_idx = None
            depth_thread.spike_neg_idx = None

        spike_detected = False
        spike_energy = 0.0

        d = delta_pixel_translation

        # -------------------------
        # 공통: 스파이크 중이면 항상 프레임 저장
        # -------------------------
        if depth_thread.spike_state != "IDLE":
            depth_thread.spike_buffer.append(
                (frame_idx, frame_pc_ts, gray_curr, depth_uint8)
            )
            if len(depth_thread.spike_buffer) > MAX_SPIKE_LEN + 2:
                depth_thread.spike_buffer.pop(0)

        # -------------------------
        # 상태 머신
        # -------------------------
        if depth_thread.spike_state == "IDLE":
            if d > PEAK_THR:
                depth_thread.spike_state = "POS_PEAK"
                depth_thread.spike_len = 0

                depth_thread.spike_pos_peak = d
                depth_thread.spike_pos_idx = frame_idx

                depth_thread.spike_buffer.clear()
                depth_thread.spike_buffer.append(
                    (frame_idx, frame_pc_ts, gray_curr, depth_uint8)
                )

        elif depth_thread.spike_state == "POS_PEAK":
            depth_thread.spike_len += 1

            if d > depth_thread.spike_pos_peak:
                depth_thread.spike_pos_peak = d
                depth_thread.spike_pos_idx = frame_idx

            if d < -PEAK_THR:
                depth_thread.spike_state = "NEG_PEAK"
                depth_thread.spike_neg_peak = d
                depth_thread.spike_neg_idx = frame_idx

            elif depth_thread.spike_len > MAX_SPIKE_LEN:
                depth_thread.spike_state = "IDLE"

        elif depth_thread.spike_state == "NEG_PEAK":
            depth_thread.spike_len += 1

            if d < depth_thread.spike_neg_peak:
                depth_thread.spike_neg_peak = d
                depth_thread.spike_neg_idx = frame_idx

            if abs(d) < NOISE_THR:
                spike_energy = abs(
                    depth_thread.spike_pos_peak -
                    depth_thread.spike_neg_peak
                )
                spike_detected = True

                # -------------------------
                # 절대화 트리거 프레임 추출
                # -------------------------
                pos_frame = None
                neg_frame = None

                pos_ts = None
                neg_ts = None

                for idx, ts, gray, depth in depth_thread.spike_buffer:
                    if idx == depth_thread.spike_pos_idx:
                        pos_frame = (gray, depth)
                        pos_ts = ts
                    elif idx == depth_thread.spike_neg_idx:
                        neg_frame = (gray, depth)
                        neg_ts = ts

                if pos_ts is not None and neg_ts is not None:
                    spike_ts = int(0.5 * (pos_ts + neg_ts))
                elif pos_ts is not None:
                    spike_ts = pos_ts
                elif neg_ts is not None:
                    spike_ts = neg_ts
                else:
                    spike_ts = None

                if pos_frame is not None and neg_frame is not None:
                    depth_thread.abs_trigger_frames = {
                        "pos": pos_frame,
                        "neg": neg_frame,
                        "energy": spike_energy,
                        "spike_ts":spike_ts
                    }

                # reset
                depth_thread.spike_state = "IDLE"
                depth_thread.spike_buffer.clear()
                depth_thread.spike_pos_idx = None
                depth_thread.spike_neg_idx = None

        # -------------------------
        # 디버그 출력
        # -------------------------
        if spike_detected:
            print(
                f"[DEPTH SPIKE] energy={spike_energy:.3f}, "
                f"len={depth_thread.spike_len}",
                f"spike_ts={spike_ts}",
                f"pulse_ts={current_pc_ts}"
            )

        prev_pixel_translation = pixel_translation

        #print(
            #f"trans_amount={trans_amount}, "
            #f"delta_pixel_translation={delta_pixel_translation}"
        #)

        # -------------------------
        # Energy → Translation mapping
        # -------------------------
        E_SMALL = 60.0    # 실험 기반 튜닝
        E_HIGH  = 140.0

        def energy_to_trans_amount(energy):
            if energy is None or energy <= 0.0:
                return 0.0
            elif energy < E_SMALL:
                return 0.04     # 4 cm
            elif energy < E_HIGH:
                return 0.08     # 8 cm
            else:
                return 0.12     # 12 cm

        trans_amount = energy_to_trans_amount(energy)
        if trans_amount is not None:
            trans_amount *= direction

        # -------------------------------------------------
        # Absolute scale trigger (energy + spike sync)
        # -------------------------------------------------
        if (
            not absolute_scale_fixed
            and hasattr(depth_thread, "abs_trigger_frames")
        ):
            spike_info = depth_thread.abs_trigger_frames

            spike_energy = spike_info["energy"]
            spike_ts     = spike_info["spike_ts"]

            if trans_amount > 0 and spike_ts is not None:
                # 시간 동기화 확인
                if abs(spike_ts - current_pc_ts) <= SYNC_WINDOW_NS:
                    print("sync confirmed")

                    gray_pos, depth_pos = spike_info["pos"]
                    gray_neg, depth_neg = spike_info["neg"]

                    diff_translation, inliers = estimate_pixel_translation_and_inliers(
                        gray_pos, gray_neg
                    )

                    if diff_translation is None or diff_translation < 1.0:
                        pass  # 실패 → 다음 기회
                    else:
                        inlier_disp, inlier_pts = inliers

                        k_abs = compute_absolute_scale_k(
                            trans_amount=trans_amount,
                            pixel_translation=diff_translation,
                            inlier_disp=inlier_disp,
                            inlier_pts=inlier_pts,
                            depth_uint8=depth_neg,   # 기준 프레임
                            gray_shape=gray_neg.shape
                        )

                        if k_abs is not None:
                            absolute_scale_fixed = True
                            print(f"[ABS SCALE FIXED] k={k_abs}")

                    # 1회성 트리거 → 제거
                    del depth_thread.abs_trigger_frames

        if absolute_scale_fixed is True:
            lidar_2d_real = k_abs * lidar_2d

            min_lidar_real_val = lidar_2d_real.min()
            max_lidar_real_val = lidar_2d_real.max()

        else:
            min_lidar_real_val = None
            max_lidar_real_val = None

        if debug is True: 
            print(
                f"trans_amount={trans_amount},"
                f"edge={edge_count}, skel={skel_count}, "
                f"ratio={skel_ratio}, "
                f"theta_var={theta_var}, "
                f"valid={valid_reason}",
                f"depth similarity={sim}",
                f"scale={scale}",
                f"[ABS] k_abs = {k_abs}",
                f"[LIDAR] min={min_lidar_real_val}, max={max_lidar_real_val}"
            )


        gray_prev = gray_curr.copy()
        prev_depth_uint8 = depth_uint8.copy()
        depth_valid_prev = depth_valid_curr
        depth_prev = depth_curr
        prev_scale = scale
        prev_valid_reason = valid_reason
        frame_idx +=1

        # =========================
        # queue push
        # =========================
        processed_queue.append((
            img,
            depth,
            lidar_2d,
            lidar_conf,
            yaw_acc,
            energy,
            S_static,
            trans_amount,
            current_pc_ts,
            k_abs,
            scale,       # 추가
            valid_reason
        ))



# --------------------------------imu 적분 스레드 ----------------------------
def imu_thread(
    gyro_queue,
    accel_queue,
    delta_yaw_queue,
    pulse_queue,
    exit_flag,
    max_dt=0.05
):
    """
    gyro_queue  : (timestamp_ns, gyro_z)
    accel_queue : (timestamp_ns, ax, ay, az)
    pulse_queue : pulse events
    """

    # -----------------------------
    # Gyro integration state
    # -----------------------------
    prev_ts = None
    accum_yaw = 0.0
    scale = 1.0
    bias  = 0.0

    # -----------------------------
    # Pulse detection params
    # -----------------------------
    NOISE_THR      = 0.08
    ACTIVE_THR     = 0.18      # 양/음 유효영역 기준

    MIN_ACTIVE_NS  = 12_000_000  # 12ms (스파이크 방지)
    MIN_NOISE_NS   = 15_000_000  # 노이즈 복귀 인정 시간

    MAX_PULSE_NS   = 1_500_000_000  # 1.5s

    MERGE_GAP_NS = 300_000_000 #300ms

    merged_pulse = None

    state = "IDLE"        # IDLE / IN_PULSE
    region = "NOISE"      # NOISE / POS / NEG

    region_enter_ts = None
    pulse_start_ts  = None

    pattern = []          # ["POS", "NEG"] or ["NEG", "POS"]

    pulse_buf = []        # (ts, ax) for peak integration

    def classify_region(ax):
        if ax >= ACTIVE_THR:
            return "POS"
        elif ax <= -ACTIVE_THR:
            return "NEG"
        else:
            return "NOISE"

    # -----------------------------
    while not exit_flag.is_set():

        # =============================
        # Gyro 처리
        # =============================
        if gyro_queue:
            ts, gyro_z = gyro_queue.popleft()

            if prev_ts is not None:
                dt = (ts - prev_ts) * 1e-9
                if 0 < dt <= max_dt:
                    delta = scale * (gyro_z - bias) * dt
                    accum_yaw += delta
                    delta_yaw_queue.append((ts, accum_yaw))

            prev_ts = ts

        # =============================
        # Accel → Pulse FSM
        # =============================
        try:
            ts, pc_ts, ax, ay, az = accel_queue.popleft()
        except IndexError:
            time.sleep(0.0005)
            continue
        
        ax_abs = abs(ax)

        #print(ts, ax)

        # -----------------------------
        curr_region = classify_region(ax)

        # -------------------------
        # REGION CHANGE DETECTION
        # -------------------------
        if curr_region != region:
            region_enter_ts = ts
            region = curr_region
        
        duration_ns = ts - region_enter_ts if region_enter_ts else 0

        # =========================
        # IDLE STATE
        # =========================
        if state == "IDLE":

            if region in ("POS", "NEG") and duration_ns >= MIN_ACTIVE_NS:
                # pulse 시작
                state = "IN_PULSE"
                pulse_start_ts = ts
                pattern = [region]
                pulse_buf = [(ts, ax)]

        # =========================
        # IN_PULSE STATE
        # =========================
        elif state == "IN_PULSE":

            pulse_buf.append((ts, ax))

            # pulse timeout 보호
            if ts - pulse_start_ts > MAX_PULSE_NS:
                state = "IDLE"
                pulse_buf.clear()
                pattern.clear()
                continue

            # --------
            # NOISE 진입
            # --------
            if region == "NOISE" and duration_ns >= (1.5 * MIN_NOISE_NS):
                pass  # 단순히 기다림

            # --------
            # 반대 peak 진입
            # --------
            elif region in ("POS", "NEG"):
                last = pattern[-1]

                if region != last and duration_ns >= MIN_ACTIVE_NS:
                    pattern.append(region)

                    # 패턴 완성?
                    if len(pattern) == 2:
                        # ---- pulse 확정 ----
                        ax_vals = [abs(a) for _, a in pulse_buf]
                        ts_vals = [t for t, _ in pulse_buf]

                        dts = [(ts_vals[i+1] - ts_vals[i]) * 1e-9
                            for i in range(len(ts_vals)-1)]

                        energy = sum(ax_vals[i] * dts[i] for i in range(len(dts)))
                        peak = max(ax_vals)

                        pulse_ts = int(
                            sum(ts_vals[i] * ax_vals[i] for i in range(len(ax_vals)))
                            / sum(ax_vals)
                        )

                        direction = 1 if pattern[0] == "POS" else -1

                        new_pulse = {
                            "ts": pulse_ts,
                            "peak": peak,
                            "energy": energy,
                            "direction": direction,
                            "pattern": pattern.copy(),
                            "start_ts": ts_vals[0],
                            "end_ts": ts_vals[-1],
                            "pc_ts": pc_ts,
                        }

                        if merged_pulse is None:
                            merged_pulse = new_pulse

                        else:
                            gap = new_pulse["start_ts"] - merged_pulse["end_ts"]

                            # -------------------------
                            # MERGE 조건
                            # -------------------------
                            if gap <= (1.5 * MERGE_GAP_NS):
                                # ---- merge 수행 ----
                                merged_pulse["end_ts"] = new_pulse["end_ts"]

                                # peak는 더 큰 쪽
                                merged_pulse["peak"] = max(
                                    merged_pulse["peak"],
                                    new_pulse["peak"]
                                )

                                merged_pulse["energy"] += new_pulse["energy"]

                                # 대표 timestamp는 가운데로
                                merged_pulse["ts"] = int(
                                    (merged_pulse["start_ts"] + merged_pulse["end_ts"]) * 0.5
                                )

                            else:
                                # ---- 이전 pulse 확정 출력 ----
                                print(
                                    "MERGED PULSE",
                                    merged_pulse["pattern"],
                                    merged_pulse["peak"],
                                    merged_pulse["energy"],
                                    merged_pulse["ts"]
                                )

                                pulse_queue.append(merged_pulse)

                                merged_pulse = new_pulse


                        # reset
                        state = "IDLE"
                        pulse_buf.clear()
                        pattern.clear()

    # =============================
    # FINAL FLUSH
    # =============================
    if merged_pulse is not None:
        pulse_queue.append(merged_pulse)
        merged_pulse = None



        
#---------------------------- 상대적 이동 및 자세변환 스레드 -------------------

def optical_flow_thread(raw_queue, delta_yaw_queue, pulse_queue, motion_queue, K=None):
    """
    Optical Flow + IMU yaw fusion
    - Rotation : Essential matrix
    - Translation : rotation-removed parallel flow
    - Scale : depth map 간 상대 비율 (depth availability 기반)
    """

    import cv2
    import numpy as np
    import math
    import time

    # =========================
    # Camera / Intrinsics
    # =========================
    downscale_size = (640, 480)
    if K is None:
        K = np.array([
            [280, 0, downscale_size[0] / 2.0],
            [0, 280, downscale_size[1] / 2.0],
            [0,   0, 1.0]
        ], dtype=np.float32)

    # =========================
    # State
    # =========================
    prev_gray = None
    prev_depth = None
    prev_yaw_acc = 0.0

    R_smoothed = np.eye(3, dtype=np.float32)
    alpha_r = 0.3

    # =========================
    # Params
    # =========================
    MAX_CORNERS = 1000
    FEATURE_QUALITY = 0.01
    MIN_DISTANCE = 7
    MIN_MATCHES = 20

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    last_pulse_pc_ts = None
    TAU = 3 #250ms
    eps = 1e-6

    W_PULSE = 0.7
    W_FLOW_PAR = 0.2
    W_FLOW_MIX = 0.1

    W_GYRO = 0.7
    W_FLOW_ROT = 0.2
    W_MIX_FLOW = 0.1

    MU_MIN = 0.15
    MU_PAR_CV = 0.32

    # pulse 관련
    pulse_consumed = True  # 초기엔 True가 안전

    # flow-event 관련
    flow_event_active = False      # 현재 flow 이벤트가 유효한가
    flow_event_consumed = True     # 소비 여부
    last_flow_event_pc_ts = None
    last_event_pc_ts = None
    pc_ts = None
    apply_translation = False
    awaiting_translation = False    # 이동량 출력 대기 상태
    event_type = None
    FLOW_TAU = 0.6   # 예: 600ms (pulse보다 짧게)
    EVENT_STATIC_TAU = 0.8
    FLOW_EVENT_THR = 0.65

    IGNORE   = 0 #상태: 모름
    PARALLEL = 1 #상태: 병렬 운동 중
    ROTATION = 2 #상태: 회전 운동 중
    MIXED    = 3 #상태: 혼합 운동 중
    STATIC   = 4 #상태: 정지

    # pulse energy thresholds (예시)
    E_SMALL  = 0.08
    E_MID    = 0.18

    # 이동량 (cm 단위 예시)
    D_SMALL = 0.04   # 4 cm
    D_MID   = 0.08   # 8 cm
    D_LARGE = 0.12   # 12 cm

    k_flow = 0.05

    direction = 0

    S_static = 0.0

    ROT_MOVE_THR = 0.02      # rad/frame 또는 rad/s (아래 설명 참고)
    MAG_MOVE_THR = 12.0      # optical flow mean magnitude (normalized)


    def R_yaw(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)

    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    print("[optical_flow] thread started")

    # =========================
    # Main loop
    # =========================
    while True:

        try:
            jpeg_bytes, depth_bytes = raw_queue.popleft()
        except IndexError:
            continue

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        depth_uint8 = np.frombuffer(depth_bytes, dtype=np.uint8)

        if depth_uint8 is None or not isinstance(depth_uint8, np.ndarray):
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, downscale_size)
        depth_small = cv2.resize(depth_uint8, downscale_size)

        event_consumed = False

        energy = 0.0

        # =========================
        # IMU yaw
        # =========================
        if delta_yaw_queue:
            _, yaw_acc = delta_yaw_queue[-1]
            delta_yaw_queue.clear()
        else:
            yaw_acc = prev_yaw_acc

        # =========================
        # Pulse data
        # =========================
        if pulse_queue:
            pulse = pulse_queue[-1]
            pulse_queue.clear()

            pulse_ts = pulse["ts"]
            peak = pulse["peak"]
            energy = pulse["energy"]
            direction = pulse["direction"]
            pattern = pulse["pattern"]
            start_ts = pulse["start_ts"]
            end_ts = pulse["end_ts"]
            pc_ts = pulse["pc_ts"]

            last_pulse_pc_ts = pc_ts

            pulse_consumed = False
            flow_event_active = False
            flow_event_consumed = True
            last_flow_event_pc_ts = None

        # =========================
        # Init
        # =========================
        if prev_gray is None:
            prev_gray = gray_small
            prev_depth = depth_small
            prev_yaw_acc = yaw_acc

            # 초기값
            init_state = 0        # STATIC 등 상수
            init_S_static = 0.0
            init_energy = 0.0
            init_direction = 0
            init_pc_ts = time.monotonic_ns()  # 안전하게 현재 시간 사용

            motion_queue.append((
                frame,
                depth_uint8,
                yaw_acc,
                init_state,
                init_S_static,
                init_energy,
                init_direction,
                init_pc_ts
            ))
            continue

        current_pc_ts = time.monotonic_ns()

        # =========================
        # Optical Flow
        # =========================
        p0 = cv2.goodFeaturesToTrack(
            prev_gray, MAX_CORNERS,
            FEATURE_QUALITY, MIN_DISTANCE
        )
        if p0 is None or len(p0) < 10:
            prev_gray, prev_depth = gray_small, depth_small
            continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray_small, p0, None, **lk_params
        )
        if p1 is None:
            prev_gray, prev_depth = gray_small, depth_small
            continue

        st = st.reshape(-1)
        idx = np.where(st == 1)[0]
        if len(idx) < MIN_MATCHES:
            prev_gray, prev_depth = gray_small, depth_small
            continue

        pts_prev = p0[idx].reshape(-1, 2)
        pts_cur  = p1[idx].reshape(-1, 2)
        flow = pts_cur - pts_prev

        # =========================
        # Flow magnitude statistics
        # =========================
        mag = np.linalg.norm(flow, axis=1)

        mean_mag = np.mean(mag)
        std_mag  = np.std(mag)

        # 변동계수 (scale invariant)
        cv_mag = std_mag / (mean_mag + eps)

        # -------------------------
        # Motion state model params
        # -------------------------

        STATE_PARAMS = {
            "parallel": {
                "mu_mean": 5.0,
                "mu_cv":   0.6,
                "sig_mean": 5.0,
                "sig_cv":   0.25,
            },
            "rotation": {
                "mu_mean": 20.0,
                "mu_cv":   0.10,
                "sig_mean": 10.0,
                "sig_cv":   0.05,
            },
            "mixed": {
                "mu_mean": 10.0,
                "mu_cv":   0.35,
                "sig_mean": 8.0,
                "sig_cv":   0.20,
            }
        }


        def gaussian_score(x, mu, sigma):
            return np.exp(-0.5 * ((x - mu) / (sigma + 1e-6))**2)


        scores = {}

        # parallel / mixed : cv only
        for state in ("parallel", "mixed"):
            p = STATE_PARAMS[state]
            scores[state] = gaussian_score(cv_mag, p["mu_cv"], p["sig_cv"])

        # rotation : mean × cv
        p = STATE_PARAMS["rotation"]
        scores["rotation"] = (
            gaussian_score(mean_mag, p["mu_mean"], p["sig_mean"]) *
            gaussian_score(cv_mag,   p["mu_cv"],   p["sig_cv"])
        )

        score_vec = np.array([
            scores["parallel"],
            scores["rotation"],
            scores["mixed"]
        ], dtype=np.float32)

        score_sum = np.sum(score_vec) + eps
        prob_parallel, prob_rotation, prob_mixed = score_vec / score_sum


        #print(
            #f"mean={mean_mag:.3f}, cv={cv_mag:.3f} | "
            #f"P(par)={prob_parallel:.2f}, "
            #f"P(rot)={prob_rotation:.2f}, "
            #f"P(mix)={prob_mixed:.2f}"
        #)

        #--------------------- 움직임 분류 ---------------------------

        delta_yaw = yaw_acc - prev_yaw_acc
        delta = abs(delta_yaw)

        E_rot = 0.0

        def rot_score_from_gyro(delta,
                                delta0=0.05,   # 2.9rad, 노이즈 상한
                                k=10.0):
            return 1.0 / (1.0 + np.exp(-k * (delta - delta0)))

        def mean_gate(mean,
                      mean0=MU_MIN,
                      k=10.0):
            # mean이 일정 이상일 때만 cv 해석 가능
            return 1.0 / (1.0 + np.exp(-k * (mean - mean0)))

        def par_score_from_cv(cv,
                      cv0=MU_PAR_CV,
                      k=8.0):
            # cv 작을수록 1, 클수록 0
            return 1.0 / (1.0 + np.exp(k * (cv - cv0)))


        E_rot += W_GYRO * rot_score_from_gyro(delta)
        E_rot += W_FLOW_ROT * prob_rotation
        E_rot += W_MIX_FLOW * prob_mixed

        E_par = 0.0

        pulse_active = (
            (last_pulse_pc_ts is not None) and
            (not pulse_consumed) and
            ((current_pc_ts - last_pulse_pc_ts) * 1e-9 <= TAU)
        )

        if pulse_active and not event_consumed:
            # ---- 분류 점수 ----
            E_par += W_PULSE
            E_par += W_FLOW_PAR * prob_parallel + W_FLOW_MIX * prob_mixed

            # ---- 이벤트 소비 ----
            pulse_consumed = True
            event_consumed = True
            event_type = "pulse"

            last_event_pc_ts = current_pc_ts
            awaiting_translation = True

        # -----------------------
        # SOFT event : flow-only
        # -----------------------
        else:
            g_mean = mean_gate(mean_mag)
            flow_score = g_mean * par_score_from_cv(cv_mag)

            # flow 이벤트 트리거
            if (
                flow_score >= FLOW_EVENT_THR and
                not flow_event_active
            ):
                flow_event_active = True
                flow_event_consumed = False
                last_flow_event_pc_ts = current_pc_ts

            # flow 이벤트가 활성 상태일 때만 E_par 부여
            if (
                flow_event_active and
                not flow_event_consumed and
                ((current_pc_ts - last_flow_event_pc_ts) * 1e-9 <= FLOW_TAU)
            ):
                E_par = flow_score
                flow_event_consumed = True   # 🔑 한 번만 의미 부여

        if flow_event_active:
            if (current_pc_ts - last_flow_event_pc_ts) * 1e-9 > FLOW_TAU:
                flow_event_active = False
                flow_event_consumed = True
                event_consumed = True
                event_type = "flow"

                # 🔑 추가
                last_event_pc_ts = current_pc_ts
                awaiting_translation = True

        if E_par >= 0.65 and E_rot < 0.65:
            state = PARALLEL
            #print("parallel")
        elif E_rot >= 0.65 and E_par < 0.65:
            state = ROTATION
            #print("rotation")
        elif E_par >= 0.65 and E_rot >= 0.65:
            state = MIXED
            #print("mixed")
        else:
            state = IGNORE
            

        #--------------------------------------------------------------
        #  STATIC GATE
        #--------------------------------------------------------------

        alpha = 0.7
        
        if delta > ROT_MOVE_THR or mean_mag > MAG_MOVE_THR:
            # 확실한 이동
            S_inst = 0.0
        else:
            # 애매하거나 미세 → 정지 취급
            S_inst = 1.0

        S_inst = np.clip(S_inst, 0.0, 1.0)

        if S_inst == 0.0:
            # 이동 시작 → 빠르게 하락
            S_static = min(S_static, 0.2)
        else:
            # 정지 유지 → 천천히 회복
            S_static = alpha * S_static + (1 - alpha) * 1.0

        #print(S_static, delta, mean_mag)

        STATIC_THR = 0.8

        if S_static >= STATIC_THR :
            state = STATIC
            #print("static")
            # yaw는 imu thread에서만 처리

        motion_queue.append((
            frame, depth_uint8, yaw_acc, state, S_static, energy, direction, current_pc_ts
        ))

        prev_gray, prev_depth = gray_small, depth_small
        prev_yaw_acc = yaw_acc



        
# --------------------절대 좌표화 및 최적 2d lidar map 선정 스레드 ------------

class SimpleKalman:
    """1D Kalman Filter for yaw smoothing"""
    def __init__(self, q=0.001, r=0.01, x0=0.0):
        self.q = q      # process noise
        self.r = r      # measurement noise
        self.x = x0     # initial state
        self.p = 1.0    # initial estimation covariance

    def update(self, measurement):
        # prediction
        self.p += self.q
        # Kalman gain
        k = self.p / (self.p + self.r)
        # update estimate
        self.x += k * (measurement - self.x)
        # update covariance
        self.p *= (1 - k)
        return self.x

# 글로벌 칼만 필터 초기화
yaw_kalman = SimpleKalman(q=0.001, r=0.01)

global_yaw = 0.0
global_x = 0.0
global_z = 0.0

def pose_update_thread(processed_queue, exit_flag):
    global global_yaw, global_x, global_z, first_frame
    global min_x, max_x, min_z, max_z, scale_factor

    first_frame = True
    traj = []
    vis_scale = 50

    canvas_size = 600
    center = canvas_size // 2
    margin = 50   # 화면 여백

    traj_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    trans_amount = 0.0
    yaw_acc = 0.0

    while not exit_flag.is_set():
        if not processed_queue:
            time.sleep(0.001)
            continue

        if first_frame:
            processed_queue.popleft()
            first_frame = False
            continue

        (img, depth, lidar_2d, lidar_conf, yaw_acc, energy, S_static, trans_amount, current_pc_ts, k_abs, scale, valid_reason) = processed_queue.popleft()

        # ================= yaw 설정 (부호 반전) =================
        global_yaw = -float(yaw_acc)
        # ================= 이동 =================
        dx_local = trans_amount
        dz_local = 0.0

        cos_y = math.cos(global_yaw)
        sin_y = math.sin(global_yaw)

        dx_global = dx_local * cos_y - dz_local * sin_y
        dz_global = dx_local * sin_y + dz_local * cos_y

        global_x += dx_global
        global_z += dz_global

        # ================= 궤적 저장 및 범위 업데이트 =================
        traj.append((global_x, global_z))

        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            zs = [p[1] for p in traj]

            min_x, max_x = min(xs), max(xs)
            min_z, max_z = min(zs), max(zs)

            span_x = max_x - min_x
            span_z = max_z - min_z

            span = max(span_x, span_z, 1e-6)
            vis_scale = (canvas_size - 2 * margin) / span

            # 궤적 그리기
            for px, pz in traj:
                x_pix = int(center + (px - min_x) * vis_scale)
                z_pix = int(center - (pz - min_z) * vis_scale)
                #cv2.circle(traj_img, (x_pix, z_pix), 2, (0, 255, 0), -1)
        else:
            vis_scale = 50.0  # 초기 기본 스케일
            px, pz = traj[0]
            x_pix = int(center + px * vis_scale)
            z_pix = int(center - pz * vis_scale)
            #cv2.circle(traj_img, (x_pix, z_pix), 2, (0, 255, 0), -1)

        # ================= grid =================
        #if first_frame:
            #traj_img[:] = 0
            #cv2.line(traj_img, (center, 0), (center, canvas_size), (70, 70, 70), 1)
            #cv2.line(traj_img, (0, center), (canvas_size, center), (70, 70, 70), 1)

        # ================= 현재 방향 화살표 =================
        #arrow_len = 30
        #arrow_x = int(x_pix + arrow_len * math.sin(global_yaw))
        #arrow_z = int(z_pix - arrow_len * math.cos(global_yaw))

        #cv2.arrowedLine(
            #traj_img,
            #(x_pix, z_pix),
            #(arrow_x, arrow_z),
            #(0, 0, 255),
            #2,
            #tipLength=0.3
        #)

        # ================= 화면 방향 Compass =================
        #compass_origin = (canvas_size - 80, 80)

        # 화면 +Z (위)
        #cv2.arrowedLine(
            #traj_img,
            #compass_origin,
            #(compass_origin[0], compass_origin[1] - 40),
            #(255, 255, 255),
            #2
        #)
        #cv2.putText(traj_img, "+Z", (compass_origin[0] - 10, compass_origin[1] - 45),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 현재 yaw 방향
        #yaw_x = int(compass_origin[0] + 40 * math.sin(global_yaw))
        #yaw_z = int(compass_origin[1] - 40 * math.cos(global_yaw))

        global_queue.append((
            img,
            lidar_2d,      # or lidar_map
            lidar_conf,
            scale,
            global_x,
            global_z,
            global_yaw,
            energy,
            S_static,
            valid_reason,
            k_abs,
            current_pc_ts
        ))

        #cv2.arrowedLine(
            #traj_img,
            #compass_origin,
            #(yaw_x, yaw_z),
            #(0, 255, 255),
            #2
        #)
        #cv2.putText(traj_img, "Yaw", (compass_origin[0] - 20, compass_origin[1] + 15),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        #cv2.imshow("Global Trajectory", traj_img)
        #cv2.waitKey(1)

# -------------------------------------- 로컬 엔트로피 계산 함수 -------------------------
def tile_entropy(tile):
    p = tile.flatten()
    p = p[p > 0.05]        # 거의 빈 공간 제거
    if len(p) < 10:
        return 0.0
    hist, _ = np.histogram(p, bins=8, range=(0,1), density=True)
    hist += 1e-6
    return -np.sum(hist * np.log(hist))

# ---------------------------------------------------
def circular_median(angles):
    angles = np.asarray(angles)
    sin_sum = np.median(np.sin(angles))
    cos_sum = np.median(np.cos(angles))
    return math.atan2(sin_sum, cos_sum)

# -------------------------------------------------------------
def view_consistency_score(pts_local, yaw_test, anchor_yaw):
    dy = yaw_test - anchor_yaw
    dy = (dy + math.pi) % (2 * math.pi) - math.pi

    # yaw 자체가 anchor에서 벗어나면 강하게 감점
    yaw_penalty = abs(dy) / math.radians(10.0)  # 10도 기준
    yaw_score = max(0.0, 1.0 - yaw_penalty)

    # 회전 후 전방 점 비율
    c = math.cos(yaw_test)
    s = math.sin(yaw_test)
    R = np.array([[c, -s],
                  [s,  c]])

    pts_rot = pts_local @ R.T

    # 전방(z > 0) + 너무 옆으로 안 퍼진 점
    forward = pts_rot[:, 1] > 0
    lateral = np.abs(pts_rot[:, 0]) < pts_rot[:, 1] * 1.2

    valid = forward & lateral

    if len(valid) == 0:
        return 0.0

    fwd_ratio = np.sum(valid) / len(valid)

    return yaw_score * fwd_ratio

#------------------------------------------------------------------------------
def raycast_cells(sx, sz, ex, ez):
    dx = abs(ex - sx)
    dz = abs(ez - sz)

    x, z = sx, sz
    sx_step = 1 if ex > sx else -1
    sz_step = 1 if ez > sz else -1

    err = dx - dz
    max_steps = dx + dz + 2

    for _ in range(max_steps):
        yield x, z

        if x == ex and z == ez:
            break

        e2 = 2 * err
        if e2 > -dz:
            err -= dz
            x += sx_step
        if e2 < dx:
            err += dx
            z += sz_step

# ---------------------------------------------------------------------
def ger_weight(d):
    GER_DIST_MIN = 1.5
    GER_DIST_MAX = 6.0
    if d < GER_DIST_MIN or d > GER_DIST_MAX:
        return 0.2      # 거의 반영 안 함
    elif d < GER_DIST_MIN + 0.3:
        return 1.5      # 최강
    elif d > GER_DIST_MAX - 0.3:
        return 1.2
    else:
        return 1.0

# ------------------- Mapping 스레드 (2D lidar map만 표시) -------------------
def mapping_thread(global_queue, corrected_queue, exit_flag):
    import numpy as np
    import cv2
    import time
    import math
    from scipy.spatial import cKDTree

    # ================== MAP PARAM ==================
    MAP_SIZE   = 3000
    RESOLUTION = 0.005
    origin     = MAP_SIZE // 2

    occ_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)  # 신뢰도 fusion 위해 float 사용

    # ================== LIDAR PARAM ==================
    LIDAR_W = 64
    half_w  = LIDAR_W / 2.0

    # ================== STATIC BUFFER ==================
    lidar_buf = []
    conf_buf  = []
    score_buf = []
    yaw_buf = []

    # ================== THRESHOLD ==================
    STATIC_GOOD     = 0.7
    STATIC_DROP     = 0.30
    CONF_PERCENTILE = 30

    ALPHA_STATIC  = 0.6   # p_static
    BETA_SPATIAL  = 0.4   # lidar_conf

    print("[mapping] multi-frame fusion thread started")

    # 전역 보정을 위한 이전 점 저장 (NumPy 배열)
    global_pts_history = np.empty((0, 2), dtype=np.float32)

    # ================== Sliding buffer for fusion + 기타 변수 설정  ==================
    fusion_buffer_size = 5  # 최근 N프레임
    fusion_pts_list = []
    fusion_conf_list = []

    trajectory = []

    prev_center_local = None
    prev_yaw = None
    prev_pts_local = None
    prev_lidar_sel = None
    prev_yaw_refined = None
    prev_best_score = None
    prev_yaw_refined_scan = None
    prev_yaw_refined_pose = None
    prev_static = None
    prev_scale = None
    disp = None
    static = True
    state = "WAIT_TRIGGER"


    prev_x = 0.0
    prev_z = 0.0

    # ===== map revise control =====
    map_update_count = 0

    # 주기 설정
    SLIMMING_PERIOD = 6        # update 6회마다 기하학적 slimming
    ENTROPY_CHECK_PERIOD = 2   # update 2회마다 entropy 평가

    # grid 분할
    TILE_SIZE = 25   # grid cell 기준 (500x500 map이면 20x20 tile)

    # 거리 기반 GER (너무 가까운 노이즈 제거)
    GER_DIST_MIN = 1
    GER_DIST_MAX = 8
    GER_RADIUS = 6

    CAMERA_HFOV_RAD = 1.48353

    ALPHA_SCALE = 0.2

    trigger_activated = False

    STATIC_THR  = 0.3

    non_static_count = 0
    STATIC_EXIT_THR = 3

    slimming = True
    reinforcement = True

    while not exit_flag.is_set():

        if not global_queue:
            time.sleep(0.001)
            continue

        try:
            frame, lidar_map, lidar_conf, scale, x, z, yaw, energy, S_static, valid_reason, k_abs, current_pc_ts = global_queue.popleft()
            #print(S_static, energy)
        except:
            continue

        lidar_map  = np.asarray(lidar_map,  dtype=np.float32).reshape(-1)
        lidar_conf = np.asarray(lidar_conf, dtype=np.float32).reshape(-1)

        if lidar_map.size != LIDAR_W:
            print("lidar map size!")
            continue

        #print("loop tick", prev_state, state)

        # --- scale accumulation (trigger 무관) ---
        if scale is not None:
            if prev_scale is None:
                total_scale = scale
            else:
                total_scale *= scale
            prev_scale = scale
        
        # --- static 판정 ---
        if S_static < STATIC_THR or energy > 0:
            static = False
        else:
            static = True

        if state == "WAIT_TRIGGER":

            if k_abs is not None and k_abs > 0:
                trigger_activated = True
                print("ABSOLUTE SCALE TRIGGERED")

                lidar_buf.clear()
                conf_buf.clear()
                score_buf.clear()
                yaw_buf.clear()

                state = "STATIC_COLLECT"

            prev_static = static
            continue

        if state == "STATIC_COLLECT":

            if static:
                print("collecting buffer")

                non_static_count = 0
                
                # ---- buffer collecting ----
                spatial_score = float(np.mean(lidar_conf))

                if S_static < 0.4:
                    static_weight = 0.6
                elif S_static < 0.6:
                    static_weight = 0.6 + (S_static - 0.45) / 0.15 * 0.3
                else:
                    static_weight = min(1.0, 0.9 + (S_static - 0.6) * 0.5)

                frame_score = spatial_score * static_weight

                lidar_buf.append(lidar_map.copy())
                conf_buf.append(lidar_conf.copy())
                score_buf.append(frame_score)
                yaw_buf.append(yaw)

            else:
                non_static_count += 1

            if non_static_count >= STATIC_EXIT_THR:
                state = "PROCESS_EVENT"
                non_static_count = 0
            
            prev_static = static
            continue

        if state == "PROCESS_EVENT":
            print("selecting buffer")

            if len(lidar_buf) < 3:
                # 데이터 부족 → 무시
                lidar_buf.clear()
                conf_buf.clear()
                score_buf.clear()
                yaw_buf.clear()
                state = "STATIC_COLLECT"
                continue

            best_idx   = int(np.argmax(score_buf))
            lidar_best = lidar_buf[best_idx]
            conf_best  = conf_buf[best_idx]
            yaw_anchor = circular_median(yaw_buf) if len(yaw_buf) >= 5 else yaw

            lb = lidar_best.astype(np.float32)

            lb_min = lb.min()
            lb_max = lb.max()
            lb_range = max(lb_max - lb_min, 1e-6)

            if scale is None:
                scale = 1
            
            conf_best  = conf_buf[best_idx]

            lidar_norm = (lb - lb_min) / lb_range

            # ================= 2.5️⃣ SELECT ANCHOR YAW =================

            if len(yaw_buf) >= 5:
                yaw_anchor = circular_median(yaw_buf)
            else:
                yaw_anchor = yaw   # fallback

            # ================= 3️⃣ POINT SELECTION (polar → local) =================
            conf_thr = max(np.percentile(conf_best, CONF_PERCENTILE), 0.18)
            pts_local, pts_conf, lidar_sel = [], [], []

            HFOV = CAMERA_HFOV_RAD   # 반드시 rad

            for i in range(LIDAR_W):
                if conf_best[i] < conf_thr:
                    continue

                r = lidar_best[i]
                if r <= 0:
                    continue

                theta = (i - half_w) / half_w * (HFOV / 2.0)

                lx = r * np.sin(theta)
                lz = r * np.cos(theta)

                pts_local.append([lx, lz])
                pts_conf.append(conf_best[i])
                lidar_sel.append(lidar_norm[i])

            pts_local = np.asarray(pts_local, dtype=np.float32)
            lidar_sel  = np.asarray(lidar_sel, dtype=np.float32)

            # ================= 3.5️⃣ YAW REFINEMENT (local scan matching) =================


            # --- yaw 탐색 범위 (event spike 대응용) ---
            YAW_RANGE = math.radians(5.0)    # ±3도
            YAW_STEP  = math.radians(0.5)    # 0.5도 간격

            yaw_candidates = np.arange(
                yaw - YAW_RANGE,
                yaw + YAW_RANGE + 1e-6,
                YAW_STEP
            )

            best_score = -1e9
            best_yaw   = yaw
            best_pts_global = None

            # --- 센서 위치 ---
            t_curr = np.array([x, z], dtype=np.float32)

            scores = []
            pts_global_candidates = []

            W_VIEW = 0.6
            W_MAP  = 0.3
            W_TEMP = 0.1

            #print("yaw test start")

            for yaw_test in yaw_candidates:

                # 회전
                c = math.cos(yaw_test)
                s = math.sin(yaw_test)
                R = np.array([[c, s],
                              [-s,  c]], dtype=np.float32)

                pts_global_test = pts_local @ R.T
                pts_global_test[:, 0] += x
                pts_global_test[:, 1] += z

                # ================= view score (주 점수) =================
                view_score = view_consistency_score(
                    pts_local, yaw_test, yaw
                )

                # ================= map score (보조) =================
                map_score = 0.0
                valid_cnt = 0

                for (gx, gz), conf in zip(pts_global_test, pts_conf):
                    mx = int(gx / RESOLUTION) + origin
                    mz = int(gz / RESOLUTION) + origin

                    if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
                        map_score += occ_grid[mz, mx] * conf
                        valid_cnt += 1

                if valid_cnt > 0:
                    map_score /= valid_cnt
                else:
                    map_score = 0.0

                # ================= temporal smoothness =================
                temp_score = 0.0
                if prev_yaw_refined_scan is not None:
                    dy = yaw_test - prev_yaw_refined_scan
                    dy = (dy + math.pi) % (2 * math.pi) - math.pi
                    temp_score = max(0.0, 1.0 - abs(dy) / YAW_RANGE)

                # ================= total score =================
                score = (
                    W_VIEW * view_score +
                    W_MAP  * map_score +
                    W_TEMP * temp_score
                )

                scores.append(score)
                pts_global_candidates.append(pts_global_test)

            #print("yaw test end")

            # --- yaw 확정 ---
            scores = np.array(scores)

            K = 3
            idx = np.argsort(scores)[-K:]

            weights = scores[idx] - scores[idx].min()
            weights = np.maximum(weights, 1e-3)

            best_idx   = idx[-1]
            best_score = scores[best_idx]

            # 기본: soft yaw
            yaw_soft = np.sum(yaw_candidates[idx] * weights) / np.sum(weights)
            yaw_refined = yaw_soft

            #if prev_best_score is not None:
                #if scores.max() < prev_best_score * 1.05:
                    #yaw_refined = prev_yaw_refined_scan

            # pts_global은 최고점 기준으로
            pts_global_yaw = pts_global_candidates[idx[-1]]

            prev_best_score = best_score
            prev_yaw_refined_scan = yaw_refined

            # ================= 4️⃣ SCALE CORRECTION (distribution-aware) =================

            scale_corr = 1.0
            total_scale = scale

            # --- anchor yaw 기준 회전 ---
            c_anchor = math.cos(yaw)
            s_anchor = math.sin(yaw)
            R_anchor = np.array([[c_anchor, s_anchor],
                                 [-s_anchor,  c_anchor]], dtype=np.float32)

            pts_global_anchor = pts_local @ R_anchor.T
            pts_global_anchor[:, 0] += x
            pts_global_anchor[:, 1] += z

            # --- 센서 기준 거리 ---
            d_anchor = np.linalg.norm(
                pts_global_anchor - np.array([x, z], dtype=np.float32),
                axis=1
            )

            mask_dist = (d_anchor > GER_DIST_MIN) & (d_anchor < GER_DIST_MAX)

            if prev_pts_local is not None and prev_lidar_sel is not None:

                # --- 거리 기반 scale 방향 ---
                r_curr = np.linalg.norm(pts_local, axis=1)
                r_prev = np.linalg.norm(prev_pts_local, axis=1)

                # --- 기본 유효성 ---
                base_valid = (r_curr > 1e-3) & (r_prev > 1e-3)

                # --- GER 제한 ---
                valid = base_valid & mask_dist
                
                if np.count_nonzero(valid) > 10:

                    scale_ratio = r_prev[valid] / r_curr[valid]
                    scale_ratio = scale_ratio[np.isfinite(scale_ratio)]

                    raw_scale = 1.0   # ⭐ default

                    if scale_ratio.size > 10:
                        raw_scale = np.median(scale_ratio)
                    else:
                        return   # 또는 scale correction 스킵


                    # --- 라이다 분포 유사도 게이트 ---
                    # lidar_norm, prev_lidar_norm은 미리 계산되어 있어야 함
                    dist_diff = np.median(np.abs(lidar_sel[valid] - prev_lidar_sel[valid]))

                    SIM_THR = 0.15   # 경험적으로 조절
                    if dist_diff < SIM_THR:

                        # --- 제한 + EMA ---
                        if abs(yaw_refined - yaw) < math.radians(1.5):

                            raw_scale = np.clip(raw_scale, 0.97, 1.03)

                            scale_corr = 1.0 + ALPHA_SCALE * (raw_scale - 1.0)
                            total_scale = scale * scale_corr

                        else:
                            total_scale = scale   # freeze


            abs_scale = total_scale * k_abs
            print(k_abs, abs_scale)
            print(min(pts_local[:,0]),max(pts_local[:,0]),min(pts_local[:,1]),max(pts_local[:,1]))
            pts_local *= abs_scale
            #print("pts_local:", len(pts_local))

            # scale 반영된 pts_local로 global 재계산
            c_refined = math.cos(yaw_refined)
            s_refined = math.sin(yaw_refined)
            R_refined = np.array([[c_refined, s_refined],
                                  [-s_refined,  c_refined]], dtype=np.float32)

            pts_global_scaled = pts_local @ R_refined.T
            pts_global = pts_global_scaled
            pts_global[:, 0] += x
            pts_global[:, 1] += z

            #print("pts_global shape:", pts_global.shape)

            # ================= 5️⃣ GLOBAL CORRECTION (event-based) =================

            pts_global_corrected = pts_global.copy()
            x_corrected, z_corrected = x, z

            # --- 센서 기준 거리 ---
            d_global = np.linalg.norm(
                pts_global - np.array([x, z], dtype=np.float32),
                axis=1
            )

            GER_mask_global = (d_global > GER_DIST_MIN) & (d_global < GER_DIST_MAX)

            # --- 현재 프레임 중심 ---
            if np.count_nonzero(GER_mask_global) > 10:
                curr_center = pts_global[GER_mask_global].mean(axis=0)
            else:
                curr_center = pts_global.mean(axis=0)

            if prev_center_local is not None and prev_yaw_refined_pose is not None:

                # yaw 변화
                dyaw = yaw_refined - prev_yaw_refined_pose
                dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

                # 역회전
                c_inv = math.cos(-dyaw)
                s_inv = math.sin(-dyaw)
                R_inv = np.array([[c_inv, s_inv],
                                  [-s_inv,  c_inv]], dtype=np.float32)

                # 현재 center를 이전 local frame으로
                curr_center_local = R_inv @ (curr_center - t_curr)

                delta = curr_center_local - prev_center_local
                dist = np.linalg.norm(delta)

                MAX_TRANS = 0.04  # meters

                if dist > MAX_TRANS:
                    delta *= (MAX_TRANS / dist)
                    curr_center_local = prev_center_local + delta

                # event-to-event translation 오차
                err_local = prev_center_local - curr_center_local

                # 다시 global
                c_dyaw = math.cos(dyaw)
                s_dyaw = math.sin(dyaw)
                R_dyaw = np.array([[c_dyaw, s_dyaw],
                                   [-s_dyaw,  c_dyaw]], dtype=np.float32)

                err_global = R_dyaw @ err_local

                # 보정 적용 (full apply)
                pts_global_corrected += err_global
                x_corrected += err_global[0]
                z_corrected += err_global[1]

                # 기준 갱신 (억제 없는 raw 기준)
                prev_center_local = curr_center_local
                prev_yaw_refined_pose = yaw_refined

            else:
                # 첫 이벤트 기준
                prev_center_local = curr_center - t_curr
                prev_yaw_refined_pose = yaw_refined

            # corrected pose 전달 (event only)
            corrected_queue.append((x_corrected, z_corrected))

            #print(delta_corrected_x, delta_corrected_z)

            mx_arr = (pts_global_corrected[:, 0] / RESOLUTION) + origin
            mz_arr = (pts_global_corrected[:, 1] / RESOLUTION) + origin

            #print(min(mx_arr), max(mx_arr), min(mz_arr), max(mz_arr))

            # ================= 6️⃣ FUSION BUFFER UPDATE =================
            fusion_pts_list.append(pts_global_corrected)
            fusion_conf_list.append(pts_conf)

            if len(fusion_pts_list) > fusion_buffer_size:
                fusion_pts_list.pop(0)
                fusion_conf_list.pop(0)

            # ================= 7️⃣ OCCUPANCY GRID UPDATE (multi-frame fusion) =================

            # 센서(카메라) 위치 → grid 좌표
            sx = int(x_corrected / RESOLUTION) + origin
            sz = int(z_corrected / RESOLUTION) + origin

            if 0 <= sx < MAP_SIZE and 0 <= sz < MAP_SIZE:

                for (gx, gz), conf in zip(pts_global_corrected, pts_conf):

                    ex = int(gx / RESOLUTION) + origin
                    ez = int(gz / RESOLUTION) + origin

                    if not (0 <= ex < MAP_SIZE and 0 <= ez < MAP_SIZE):
                        continue

                    # Bresenham line algorithm
                    dx = abs(ex - sx)
                    dz = abs(ez - sz)
                    x0, z0 = sx, sz
                    sx_step = 1 if ex > sx else -1
                    sz_step = 1 if ez > sz else -1
                    err = dx - dz

                    loop_guard = 0

                    for x0, z0 in raycast_cells(sx, sz, ex, ez):

                        if not (0 <= x0 < MAP_SIZE and 0 <= z0 < MAP_SIZE):
                            break

                        occ_grid[z0, x0] *= 0.995
                
            # ---- 맵 전체에 아주 약한 decay ----
            #occ_grid *= 0.999   # 0.995~0.999 사이 추천

            # ---- 새로운 관측 반영 ----
            for i, (gx, gz) in enumerate(pts_global_corrected):
                mx = int(gx / RESOLUTION) + origin
                mz = int(gz / RESOLUTION) + origin

                if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
                    conf = pts_conf[i]
                    d = np.linalg.norm([gx - x_corrected, gz - z_corrected])
                    # 예시
                    # 센서 → 점 벡터
                    vx = gx - x_corrected
                    vz = gz - z_corrected

                    # 센서 전방 벡터 (yaw_refined 기준)
                    fx = math.sin(yaw_refined)
                    fz = math.cos(yaw_refined)

                    # 각도 계산 
                    dot = vx * fx + vz * fz
                    norm_v = math.hypot(vx, vz)

                    if norm_v < 1e-3:
                        continue

                    cos_angle = dot / norm_v
                    angle = math.acos(np.clip(cos_angle, -1.0, 1.0))

                    # GER 조건
                    if  angle < HFOV / 2 and scale * GER_DIST_MIN < norm_v < scale * GER_DIST_MAX:
                        w_ger = 1.0
                    elif angle < HFOV * 0.75:
                        w_ger = 0.6
                    else:
                        w_ger = 0.2

                    if w_ger > 0.9:
                        occ_grid[mz, mx] += conf * 0.6   # 🔥 강하게
                    else:
                        occ_grid[mz, mx] += conf * w_ger * 0.2

                    occ_grid[mz, mx] = min(1.0, occ_grid[mz, mx])

            # === display 좌표 스케일 ===
            scale_disp = 600 / MAP_SIZE
            px = int(sx * scale_disp)
            pz = int(sz * scale_disp)

            # === map update count ===
            map_update_count += 1
            print(map_update_count)

            lidar_buf.clear()
            conf_buf.clear()
            score_buf.clear()

            # ---------------------------------- local entropy guided geometric slimming -----------------------------

            if slimming is True and map_update_count % SLIMMING_PERIOD == 0:

                for ty in range(0, MAP_SIZE, TILE_SIZE):
                    for tx in range(0, MAP_SIZE, TILE_SIZE):

                        tile = occ_grid[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
                        if tile.size == 0:
                            continue

                        # ================= entropy 계산 =================
                        H = tile_entropy(tile)

                        # ================= entropy → 정책 =================
                        if H < 1.1:          # STABLE
                            dist_thr = 1.5
                            slim_factor = 0.9     # 거의 안 건드림

                        elif H < 1.6:        # MID
                            dist_thr = 1.2
                            slim_factor = 0.7

                        else:                # CHAOTIC 🔥 핵심
                            dist_thr = 1.0
                            slim_factor = 0.4     # 강하게 제거

                        # ================= seed 추출 =================
                        ys, xs = np.where(tile > 0.2)
                        if len(xs) < TILE_SIZE * 0.4:
                            continue

                        pts = np.stack([xs, ys], axis=1).astype(np.float32)

                        # ================= PCA =================
                        mean = pts.mean(axis=0)
                        pts_c = pts - mean
                        cov = np.cov(pts_c.T)

                        eigvals, eigvecs = np.linalg.eig(cov)
                        idx = np.argmax(eigvals)

                        # line-like 조건 완화
                        if eigvals[idx] / (eigvals.sum() + 1e-6) < 0.5:
                            continue

                        main_dir = eigvecs[:, idx]

                        # ================= 전체 점 =================
                        all_ys, all_xs = np.where(tile > 0.2)
                        all_pts = np.stack([all_xs, all_ys], axis=1).astype(np.float32)

                        pts_c_all = all_pts - mean
                        proj_all = pts_c_all @ main_dir
                        recon_all = np.outer(proj_all, main_dir)
                        dist_all = np.linalg.norm(pts_c_all - recon_all, axis=1)

                        # ================= 파라미터 =================
                        reinforce_dist = dist_thr * 0.5
                        reinforce_gain = 1.05

                        sigma = dist_thr * 0.5
                        R = max(1, int(3 * sigma))
                        splat_gain = 0.02

                        align_thr = dist_thr * 1.2
                        align_gain = 0.8

                        # ================= 1️⃣ Slimming =================
                        for (x, y), d in zip(all_pts.astype(int), dist_all):

                            if d > dist_thr:
                                tile[y, x] *= slim_factor
                                if tile[y, x] < 0.05:
                                    tile[y, x] = 0.0

                        # ================= 2️⃣ Reinforce + Splat =================
                        for (x, y), d in zip(all_pts.astype(int), dist_all):

                            if d < reinforce_dist and reinforcement:

                                tile[y, x] = min(tile[y, x] * reinforce_gain, 1.0)

                                for dy in range(-R, R + 1):
                                    for dx in range(-R, R + 1):

                                        ny, nx = y + dy, x + dx

                                        if not (0 <= ny < tile.shape[0] and 0 <= nx < tile.shape[1]):
                                            continue

                                        dd = dx * dx + dy * dy
                                        w = math.exp(-dd / (2 * sigma * sigma))

                                        tile[ny, nx] = min(
                                            tile[ny, nx] + w * splat_gain,
                                            1.0
                                        )

                        # ================= 3️⃣ Alignment (seed 포함) =================
                        for (x, y), d in zip(all_pts.astype(int), dist_all):

                            if d < align_thr:

                                p = np.array([x, y], dtype=np.float32)

                                v = p - mean
                                t = v @ main_dir
                                proj = mean + t * main_dir

                                # seed 여부에 따른 gain 조절
                                if tile[y, x] > 0.6:
                                    gain = align_gain * 0.5
                                else:
                                    gain = align_gain

                                new_p = p * (1 - gain) + proj * gain

                                nx, ny = np.round(new_p).astype(int)

                                if 0 <= ny < tile.shape[0] and 0 <= nx < tile.shape[1]:

                                     val = tile[y, x] * gain

                                     tile[y, x] *= (1 - gain)
                                     tile[ny, nx] = min(tile[ny, nx] + val, 1.0)


            # ================= Display =================
            disp = cv2.resize((occ_grid * 4 * 255).astype(np.uint8), (600, 600), interpolation=cv2.INTER_NEAREST)
            disp = cv2.dilate(disp, np.ones((3, 3), np.uint8))

            # === 1️⃣ 현재 위치 (빨간 점) ===
            if 0 <= px < 600 and 0 <= pz < 600:
                cv2.circle(disp, (px, pz), 4, (0, 0, 255), -1)

            # === 2️⃣ 방향 표시 (yaw, 파란 화살표) ===
            arrow_len = 14
            dx = int(math.sin(yaw_refined) * arrow_len)
            dz = int(math.cos(yaw_refined) * arrow_len)

            cv2.arrowedLine(
                disp,
                (px, pz),
                (px + dx, pz + dz),
                (255, 0, 0),
                1,
                tipLength=0.3
            )

            # === 3️⃣ (선택) 이동 궤적 ===
            trajectory.append((px, pz))
            if len(trajectory) > 300:
                trajectory.pop(0)

            for tx, tz in trajectory:
                cv2.circle(disp, (tx, tz), 1, (0, 255, 255), -1)

            # ================= 8️⃣ BUFFER & HISTORY RESET =================
            global_pts_history = np.vstack([global_pts_history, pts_global_corrected])
            if len(global_pts_history) > 5000:
                global_pts_history = global_pts_history[-5000:]

            prev_pts_local = pts_local.copy()
            prev_lidar_sel = lidar_sel.copy()

            cv2.imshow("Occupancy Grid Fusion", disp)
            cv2.waitKey(1)

            continue

        prev_static = static

    cv2.destroyAllWindows()



def main():

    trajectory = []                     # trajectory list
    lock = threading.Lock()             # pose lock
    exit_flag = threading.Event()       # exit signal for threads

    pose_of = np.eye(4, dtype=np.float32)
    pose_imu = np.eye(4, dtype=np.float32)
    pose_fusion = np.eye(4, dtype=np.float32)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST,PORT))
        s.listen(1)
        print(f"Listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        print(f"Client connected: {addr}")

        # PC에서 보여줄 화면 크기
        target_H, target_W = 480, 640

        threading.Thread(target=receive_thread,args=(conn, raw_queue, gyro_queue, accel_queue),daemon=True).start()
        threading.Thread(target=imu_thread, args=( gyro_queue, accel_queue, delta_yaw_queue, pulse_queue, exit_flag,),daemon=True).start()
        threading.Thread(target=optical_flow_thread, args=(raw_queue, delta_yaw_queue, pulse_queue, motion_queue,), daemon=True).start()
        threading.Thread(target=depth_thread,daemon=True).start()
        threading.Thread(target=pose_update_thread, args=(processed_queue,exit_flag,), daemon=True).start()

        threading.Thread(target=mapping_thread,args=(global_queue, corrected_queue, exit_flag), daemon=True).start()
        

        # keep main alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__=="__main__":
    main()

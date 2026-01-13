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

HOST = '0.0.0.0'
PORT = 5000

EXPECTED_W = 256
EXPECTED_H = 256

raw_queue = deque(maxlen=1)
processed_queue = deque(maxlen=1)
motion_queue = deque(maxlen=1)
global_queue = deque(maxlen=1)
gyro_queue  = deque(maxlen=50)
delta_yaw_queue = deque(maxlen=10)
corrected_queue = deque(maxlen=1)

# ------------------- TCP 수신 -------------------
def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_thread(sock, raw_queue, gyro_queue):
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
            r0 = lidar_1d[i] + 1e-6
            cont[i] = (
                abs(lidar_1d[i] - lidar_1d[i-1]) +
                abs(lidar_1d[i] - lidar_1d[i+1])
            ) / r0

    sigma_cont = np.percentile(cont[cont > 0], 80) + 1e-6
    conf_cont = np.exp(-cont / sigma_cont)

    # slope smoothness
    grad = np.diff(lidar_1d)
    curv = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        if valid[i-1] and valid[i] and valid[i+1]:
            curv[i] = abs(grad[i] - grad[i-1]) / (lidar_1d[i] + 1e-6)

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

    V_THR = 0.4
    DEPTH_SIM_THR = 0.7
    FRAME_SIM_THR = 0.6
    diff_thr = 15
    MIN_RATIO_PIXELS = 80

    min_depth = 15
    max_depth = 240

    def clamp01(x):
        return max(0.0, min(1.0, float(x)))

    while True:
        if not raw_queue:
            time.sleep(0.001)
            continue

        jpeg_bytes, depth_bytes = raw_queue.popleft()

        depth_array = np.frombuffer(depth_bytes, dtype='<f4').copy()
        if depth_array.size != EXPECTED_W * EXPECTED_H:
            continue

        depth_frame = depth_array.reshape((EXPECTED_H, EXPECTED_W))
        depth_small = cv2.resize(depth_frame, (128, 128), interpolation=cv2.INTER_AREA)
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
        h, w = depth_small.shape
        y_positions = np.full(w, -1, dtype=np.int32)

        for x in range(w):
            column = depth_small[:, x]
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
        # lidar_1d (polar range)
        # =========================
        lidar_1d = np.zeros(w, dtype=np.float32)

        k = 100000.0        # scale (튜닝)
        eps_1 = 1e-3     # 안정화용

        for x in range(w):
            ys = y_smooth[x]
            if 0 <= ys < h:
                d = float(depth_small[ys, x])   # raw MiDaS depth
                lidar_1d[x] = k / (d + eps_1)

        #print(lidar_1d.min(),lidar_1d.max())

        # =========================
        # spatial confidence (이미 존재)
        # =========================
        lidar_conf = compute_spatial_confidence(lidar_1d)

        # =========================
        # ⭐ depth validity (scale gate) ⭐
        # =========================
        # ---- (1) depth gradient ----
        gx = cv2.Sobel(depth_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_uint8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)

        # ---- (2) strong gradient mask (노이즈 제거) ----
        thr = np.percentile(grad_mag, 85)
        mask = grad_mag > thr

        ys, xs = np.where(mask)

        # ---- (3) spatial dispersion → v_grad ----
        if len(xs) < 30:
            v_grad = 0.0
        else:
            cx = xs.mean()
            cy = ys.mean()
            var = np.mean((xs - cx) ** 2 + (ys - cy) ** 2)

            # 정규화 (해상도 불변)
            max_var = 0.25 * depth_uint8.size
            v_grad = 1.0 - np.clip(var / max_var, 0.0, 1.0)

        v_grad = clamp01(v_grad)

        # ---- (4) 기존 품질 지표 ----
        std_norm = np.std(depth_uint8) / 255.0

        if prev_depth_uint8 is None:
            temporal_diff = 0.0
        else:
            temporal_diff = np.mean(
                np.abs(
                    depth_uint8.astype(np.float32) -
                    prev_depth_uint8.astype(np.float32)
                )
            ) / 255.0

        v_std  = clamp01((std_norm - 0.05) / 0.15)
        v_temp = clamp01(1.0 - temporal_diff / 0.2)

        # ---- (5) ⭐ 최종 depth validity (선형 결합) ⭐
        depth_validity = (
            0.5 * v_grad +
            0.3 * v_std +
            0.2 * v_temp
        )

        H, W = depth_uint8.shape
        G = 3   # grid size

        diag = np.sqrt(W*W + H*H)
        SIGMA_C = 0.08*diag
        SIGMA_S = 0.01*(W*W + H*H)

        edge_mask = grad_mag > np.percentile(grad_mag, 75)

        valid_mask = (
            (depth_uint8 > min_depth)
            & edge_mask
        )

        valid_mask_curr = valid_mask

        if valid_mask_prev is None:
            depth_similarity = 0.0
            centroid_sim = 0.0
            shape_sim = 0.0
            spatial_hist_sim = 0.0
        else:
            ys, xs = np.where(valid_mask_curr)
            ys_p, xs_p = np.where(valid_mask_prev)

            # centroid
            cy, cx = ys.mean(), xs.mean()
            cy_p, cx_p = ys_p.mean(), xs_p.mean()

            centroid_dist = np.sqrt((cx - cx_p)**2 + (cy - cy_p)**2)
            centroid_sim = np.exp(-centroid_dist / SIGMA_C)

            # shape
            var_x, var_y = np.var(xs), np.var(ys)
            var_xp, var_yp = np.var(xs_p), np.var(ys_p)

            shape_sim = np.exp(
                - (abs(var_x - var_xp) + abs(var_y - var_yp)) / SIGMA_S
            )

            # spatial histogram
            spatial_hist_sim = compute_spatial_hist_sim(
                valid_mask_curr, valid_mask_prev
            )

            depth_similarity = (
                0.4 * centroid_sim +
                0.3 * shape_sim +
                0.3 * spatial_hist_sim
            )

        scale = 1.0

        depth_curr = depth_small

        # 1️⃣ bytes → numpy buffer
        np_buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)

        # 2️⃣ JPEG decode
        img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)

        # 3️⃣ BGR → Gray
        gray_curr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        if prev_scale is not None:

            if depth_validity < V_THR:
                 # (A) 노이즈
                 scale = prev_scale

            else:
                if depth_similarity > DEPTH_SIM_THR:
                     # (B) 동일 물체
                     overlap_mask = valid_mask_prev & valid_mask_curr

                else:
                    if gray_prev is not None:
                         # depth는 유효하지만 다른 물체 가능
                         diff = cv2.resize(np.abs(gray_curr - gray_prev), (H, W), interpolation=cv2.INTER_AREA)

                         same_mask = (
                             (diff < diff_thr) &
                             valid_mask_prev &
                             valid_mask_curr
                         )

                         frame_similarity = same_mask.sum() / (
                             (valid_mask_prev & valid_mask_curr).sum() + eps
                         )

                         if frame_similarity > FRAME_SIM_THR:
                             # (C) 시야 연속
                             overlap_mask = same_mask
                         else:
                             # (D) 단절
                             scale = prev_scale
                             overlap_mask = None

                    else:
                        scale = prev_scale
                        overlap_mask = None

                if overlap_mask is not None:
                     d_prev = depth_prev[overlap_mask]
                     d_curr = depth_curr[overlap_mask]

                     ratio = d_curr / (d_prev + eps)
                     ratio = ratio[(ratio > 0.3) & (ratio < 3.0)]

                     if ratio.size > MIN_RATIO_PIXELS:
                         scale = np.median(ratio)
                     else:
                         scale= prev_scale

        else:
            prev_scale = scale


        #print(depth_validity, scale, depth_similarity, depth_uint8.mean())

        gray_prev = gray_curr.copy()
        prev_depth_uint8 = depth_uint8.copy()
        valid_mask_prev = valid_mask_curr.copy()
        depth_prev = depth_curr.copy()

        # =========================
        # queue push
        # =========================
        processed_queue.append((
            jpeg_bytes,
            depth_uint8,
            lidar_1d,
            lidar_conf,
            depth_validity,   # ← ⭐ 추가된 스케일 비교 가능 지수 ⭐
            scale
        ))



# --------------------------------imu 적분 스레드 ----------------------------
def imu_thread(
    gyro_queue,
    delta_yaw_queue,
    exit_flag,
    max_dt=0.05
):
    """
    gyro_queue      : (timestamp_ns, gyro_z [rad/s])
    delta_yaw_queue : (timestamp_ns, accum_yaw [rad])
    """

    prev_ts = None
    accum_yaw = 0.0

    scale = 1
    bias = 0.0

    while not exit_flag.is_set():

        if not gyro_queue:
            time.sleep(0.001)
            continue

        ts, gyro_z = gyro_queue.popleft()

        # 첫 샘플
        if prev_ts is None:
            prev_ts = ts
            continue

        # Δt 계산 (ns → sec)
        dt = (ts - prev_ts) * 1e-9
        prev_ts = ts

        # 비정상 dt 제거
        if dt <= 0 or dt > max_dt:
            continue

        # -----------------------------
        # 적분 (rad)
        # -----------------------------
        delta = scale * (gyro_z - bias) * dt
        accum_yaw += delta

        # -----------------------------
        # OF용 누적 yaw 전달
        # -----------------------------
        delta_yaw_queue.append(
            (ts, accum_yaw)
        )

        # 디버그 출력
        #print(f"yaw_acc: {accum_yaw:+.6f} rad")
       
        
#---------------------------- 상대적 이동 및 자세변환 스레드 -------------------

def local_pose_thread(processed_queue, delta_yaw_queue, motion_queue, K=None):
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

    M_screen = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0,  1, 0],
        [0,  0,  0, 1]
    ], dtype=np.float32)

    # =========================
    # State
    # =========================
    prev_gray = None
    prev_depth = None
    prev_ratio = 1.0
    prev_yaw_acc = 0.0

    R_smoothed = np.eye(3, dtype=np.float32)
    alpha_r = 0.3

    flow_ema = None
    yaw_ema  = None
    prev_yaw_of = None

    EMA_ALPHA = 0.5   # 반응성 ↑ = 0.3~0.5, 안정성 ↑ = 0.1~0.2


    # =========================
    # Params
    # =========================
    MAX_CORNERS = 1000
    FEATURE_QUALITY = 0.01
    MIN_DISTANCE = 7
    MIN_MATCHES = 20

    MIN_VALID = 0.55
    FULL_VALID = 0.75

    FLOW_STATIC_ON  = 0.14   # state=0 진입
    FLOW_STATIC_OFF = 0.18   # state=1 복귀

    FLOW_FAST_ON = 0.9     # pixel 단위 (downscale 기준)
    YAW_FAST_ON  = 0.02    # rad (≈ 1.1도)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    state = -1
    motion_score = 0.0

    POSE_GAIN = 0.05

    def R_yaw(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]], dtype=np.float32)


    print("[optical_flow] thread started (depth-ratio based mapping)")

    # =========================
    # Main loop
    # =========================
    while True:
        if not processed_queue:
            time.sleep(0.001)
            continue

        jpeg_bytes, depth_uint8, lidar_1d, lidar_conf, depth_availability, depth_ratio = processed_queue.popleft()

        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, downscale_size)
        depth_small = cv2.resize(depth_uint8, downscale_size)

        # =========================
        # IMU yaw
        # =========================
        if delta_yaw_queue:
            _, yaw_acc = delta_yaw_queue[-1]
            delta_yaw_queue.clear()
        else:
            yaw_acc = prev_yaw_acc

        # =========================
        # Init
        # =========================
        if prev_gray is None:
            prev_gray = gray_small
            prev_depth = depth_small
            prev_yaw_acc = yaw_acc

            motion_queue.append((
                frame, depth_uint8, lidar_1d, lidar_conf,
                np.eye(4, dtype=np.float32),
                yaw_acc, 1, 0.0, prev_ratio
            ))
            continue

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
        # Rotation (Essential)
        # =========================
        yaw_of = 0.0
        E, _ = cv2.findEssentialMat(
            pts_prev, pts_cur, K,
            cv2.RANSAC, 0.999, 1.0
        )
        if E is not None:
            _, R_est, _, mask = cv2.recoverPose(E, pts_prev, pts_cur, K)
            if mask is not None and mask.sum() > 8:
                R_smoothed = (1 - alpha_r) * R_smoothed + alpha_r * R_est

        yaw_of = math.atan2(R_smoothed[1, 0], R_smoothed[0, 0])

        # =========================
        # Static 판단 (EMA)
        # =========================
        flow_mag = np.linalg.norm(flow, axis=1) 
        flow_score = np.percentile(flow_mag, 70)

        if prev_yaw_acc is None:
            yaw_delta = 0.0
        else:
            yaw_delta = abs(yaw_acc - prev_yaw_acc)
            yaw_delta = (yaw_delta + math.pi) % (2 * math.pi) - math.pi
            yaw_delta = abs(yaw_delta)

        # =========================
        # Fast trigger metrics
        # =========================
        flow_fast = np.percentile(flow_mag, 90)   # 순간적인 큰 움직임
        yaw_fast  = yaw_delta                     # raw yaw delta

        # --- EMA update ---
        if flow_ema is None:
            flow_ema = flow_score
            yaw_ema  = yaw_delta
        else:
            flow_ema = (1 - EMA_ALPHA) * flow_ema + EMA_ALPHA * flow_score
            yaw_ema  = (1 - EMA_ALPHA) * yaw_ema  + EMA_ALPHA * yaw_delta

        # =========================
        # State decision (FAST + EMA)
        # =========================

        # --- FAST trigger (즉시 moving) ---
        if state != 1 and (flow_fast > 0.9 or yaw_fast > 0.02):
            state = 1

        # --- EMA based (안정적인 복귀 판단) ---
        elif state != 0 and flow_ema < 0.40 and yaw_ema < 0.003:
            state = 0


        motion_score = flow_ema  # 디버깅용
        prev_yaw_of = yaw_of

        #print(state, flow_fast, yaw_fast, flow_ema, yaw_ema)

        ALPHA_MOTION_MIN  = 0.25   # 이 이하면 흔들림
        ALPHA_MOTION_FULL = 0.60   # 이 이상이면 확실한 이동

        alpha_motion = np.clip(
            (motion_score - ALPHA_MOTION_MIN) /
            (ALPHA_MOTION_FULL - ALPHA_MOTION_MIN),
            0.0, 1.0
        )

        YAW_ALPHA_OFF = 0.015   # rad ≈ 0.85°
 
        alpha_yaw = np.clip(
           1.0 - (yaw_ema / YAW_ALPHA_OFF),
           0.0, 1.0
        )

        alpha = alpha_motion * alpha_yaw
        #print(alpha)

        # =========================
        # Rotation 제거 → Parallel flow
        # =========================
        # 1. pixel → normalized camera coord
        pts_prev_norm = cv2.undistortPoints(
            pts_prev.reshape(-1,1,2), K, None
        ).reshape(-1,2)

        pts_prev_h = np.hstack([pts_prev_norm, np.ones((len(pts_prev_norm),1))])

        # 2. rotation
        pts_rot = (R_smoothed @ pts_prev_h.T).T

        # 3. back to normalized
        pts_rot_norm = pts_rot[:, :2] / pts_rot[:, 2:3]

        # 4. normalized → pixel
        pts_rot_pix = (K[:2,:2] @ pts_rot_norm.T).T + K[:2,2]

        flow_rot = pts_rot_pix - pts_prev
        flow_parallel = flow - flow_rot
        
        #---------------------- soft gate ---------------------------
        flow_norm = np.linalg.norm(flow_parallel, axis=1)

        FLOW_MIN  = 0.15   # 이 이하면 무시
        FLOW_FULL = 0.50   # 이 이상은 full

        w = np.clip(
            (flow_norm - FLOW_MIN) / (FLOW_FULL - FLOW_MIN),
            0.0, 1.0
        )  


        # =========================
        # Depth ratio (scale)
        # =========================
        u = pts_prev[:, 0].astype(int)
        v = pts_prev[:, 1].astype(int)
        valid = (
            (u >= 0) & (u < downscale_size[0]) &
            (v >= 0) & (v < downscale_size[1])
        )

        if np.count_nonzero(valid) > 10:
             d_prev = prev_depth[v[valid], u[valid]].astype(np.float32) + 1e-6
             d_curr = depth_small[v[valid], u[valid]].astype(np.float32) + 1e-6
             scale_meas = np.median(d_prev / d_curr)
        else:
             scale_meas = prev_ratio

        # depth validity → update gain
        if depth_availability <= MIN_VALID:
             gain = 0.0
        elif depth_availability >= FULL_VALID:
             gain = 1.0
        else:
             gain = (depth_availability - MIN_VALID) / (FULL_VALID - MIN_VALID)

        scale = (1 - gain) * prev_ratio + gain * scale_meas
        prev_ratio = scale

        #----------------------- translation -------------------------------
        if np.sum(w) > 1e-6:
            dx = -np.sum(w * flow_parallel[:,0]) / np.sum(w) / K[0,0] * scale * POSE_GAIN
            dy = -np.sum(w * flow_parallel[:,1]) / np.sum(w) / K[1,1] * scale * POSE_GAIN
        else:
            dx = dy = 0.0

        dz =  0

        if state == 0:
            dx = dy = 0.0
            
        #print(dx,dy)

        # =========================
        # Transform
        # =========================
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_yaw(yaw_acc)
        T[:3,  3] = np.array([dx, dy, dz])
        T = M_screen @ T @ M_screen
        #print(dz, scale)

        motion_queue.append((
            frame, depth_uint8, lidar_1d, lidar_conf,
            T, yaw_acc, state, alpha, depth_ratio, motion_score
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

def global_pose_thread(motion_queue, global_queue, exit_flag):
    global global_yaw, global_x, global_z, first_frame
    global min_x, max_x, min_z, max_z, scale_factor

    first_frame = True
    traj = []
    scale = 50

    present_depth_scale = 1.0
    ALPHA_SCALE = 0.05

    canvas_size = 600
    center = canvas_size // 2
    margin = 50   # 화면 여백

    traj_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    while not exit_flag.is_set():
        if not motion_queue:
            time.sleep(0.001)
            continue

        if first_frame:
            motion_queue.popleft()
            first_frame = False
            continue

        frame, depth_uint8, lidar_1d, lidar_conf, T_disp_filtered, yaw_acc, state, alpha, depth_scale, motion_score = motion_queue.popleft()
        #print(alpha)

        present_depth_scale *= (1 + ALPHA_SCALE * (depth_scale - 1))

        # ================= yaw 설정 (부호 반전) =================
        global_yaw = -float(yaw_acc)
        #print(f"[POSE] yaw_acc={yaw_acc:+.6f} | global_yaw={global_yaw:+.6f}")

        # ================= 이동 =================
        if state != 0:
            dz_local = scale * 0.005
            dx_local = 0.0

            cos_y = math.cos(global_yaw)
            sin_y = math.sin(global_yaw)

            dx_global = dx_local * cos_y - dz_local * sin_y
            dz_global = dx_local * sin_y + dz_local * cos_y

            global_x += -dx_global
            global_z += -dz_global

            # ====== mapping feedback (즉각 보정) ======
            if corrected_queue:
                x_corr, z_corr = corrected_queue.popleft()

                dx = x_corr - global_x
                dz = z_corr - global_z

                # 안전장치
                MAX_CORR = 0.3
                norm = math.hypot(dx, dz)
                if norm > MAX_CORR:
                    dx *= MAX_CORR / norm
                    dz *= MAX_CORR / norm

                # 정지/이동에 따른 gain
                GAIN = 0.5 if state == 0 else 0.2

                global_x += GAIN * dx
                global_z += GAIN * dz

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
            scale = (canvas_size - 2 * margin) / span
        else:
            scale = 50.0  # 초기 기본 스케일

        # ================= 화면 좌표 =================
        x_pix = int(center + global_x * scale)
        z_pix = int(center - global_z * scale)

        # ================= grid =================
        traj_img[:] = 0
        cv2.line(traj_img, (center, 0), (center, canvas_size), (70, 70, 70), 1)
        cv2.line(traj_img, (0, center), (canvas_size, center), (70, 70, 70), 1)

        # ================= 궤적 점 =================
        cv2.circle(traj_img, (x_pix, z_pix), 3, (0, 255, 0), -1)

        # ================= 현재 방향 화살표 =================
        arrow_len = 30
        arrow_x = int(x_pix + arrow_len * math.sin(global_yaw))
        arrow_z = int(z_pix - arrow_len * math.cos(global_yaw))

        cv2.arrowedLine(
            traj_img,
            (x_pix, z_pix),
            (arrow_x, arrow_z),
            (0, 0, 255),
            2,
            tipLength=0.3
        )

        # ================= 화면 방향 Compass =================
        compass_origin = (canvas_size - 80, 80)

        # 화면 +Z (위)
        cv2.arrowedLine(
            traj_img,
            compass_origin,
            (compass_origin[0], compass_origin[1] - 40),
            (255, 255, 255),
            2
        )
        cv2.putText(traj_img, "+Z", (compass_origin[0] - 10, compass_origin[1] - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 현재 yaw 방향
        yaw_x = int(compass_origin[0] + 40 * math.sin(global_yaw))
        yaw_z = int(compass_origin[1] - 40 * math.cos(global_yaw))

        #print(global_x, global_z)

        global_queue.append((
            frame,
            lidar_1d,      # or lidar_map
            lidar_conf,
            present_depth_scale,
            global_x,
            global_z,
            global_yaw,
            state,
            alpha,
            motion_score
        ))

        cv2.arrowedLine(
            traj_img,
            compass_origin,
            (yaw_x, yaw_z),
            (0, 255, 255),
            2
        )
        cv2.putText(traj_img, "Yaw", (compass_origin[0] - 20, compass_origin[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        #cv2.imshow("Global Trajectory", traj_img)
        cv2.waitKey(1)

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
    RESOLUTION = 0.5
    origin     = MAP_SIZE // 2

    occ_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)  # 신뢰도 fusion 위해 float 사용

    # ================== LIDAR PARAM ==================
    LIDAR_W = 128
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

    prev_state = None
    prev_center_local = None
    prev_yaw = None
    prev_pts_local = None
    prev_lidar_norm = None
    prev_yaw_refined = None
    prev_best_score = None
    prev_yaw_refined_scan = None
    prev_yaw_refined_pose = None


    prev_x = 0.0
    prev_z = 0.0
    dx = 0.0
    dz = 0.0

    s_gain = 0.1

    MOVING_CORR_EVERY = 2
    MOVING_CORR_GAIN  = 0.3
    DRIFT_CLAMP_MOVING = 2

    moving_corr_cnt = 0

    # ===== map revise control =====
    map_update_count = 0

    # 주기 설정
    SLIMMING_PERIOD = 6        # update 6회마다 기하학적 slimming
    ENTROPY_CHECK_PERIOD = 2   # update 2회마다 entropy 평가

    # grid 분할
    TILE_SIZE = 25   # grid cell 기준 (500x500 map이면 20x20 tile)

    # 거리 기반 GER (너무 가까운 노이즈 제거)
    GER_DIST_MIN = 220.0
    GER_DIST_MAX = 650.0
    GER_RADIUS = 300.0

    CAMERA_HFOV_RAD = 1.48353

    ALPHA_SCALE = 0.2

    while not exit_flag.is_set():

        if not global_queue:
            time.sleep(0.001)
            continue

        try:
            frame, lidar_map, lidar_conf, scale, x, z, yaw, state, alpha, motion_score = global_queue.popleft()
            #print(alpha)
        except:
            continue

        lidar_map  = np.asarray(lidar_map,  dtype=np.float32).reshape(-1)
        lidar_conf = np.asarray(lidar_conf, dtype=np.float32).reshape(-1)

        if lidar_map.size != LIDAR_W:
            print("lidar map size!")
            continue

        #print("loop tick", prev_state, state)

        # ================= 1️⃣ BUFFER ACCUMULATION =================

        if prev_state is None:
            prev_state = state
            continue
            
        elif state == 0:
            print("buffer collecting")

            spatial_score = float(np.mean(lidar_conf))
            frame_score   = ALPHA_STATIC * motion_score + BETA_SPATIAL * spatial_score

            lidar_buf.append(lidar_map.copy())
            conf_buf.append(lidar_conf.copy())
            score_buf.append(frame_score)
            yaw_buf.append(yaw)

            prev_state = state
            continue
            
        # ================= 2️⃣ SELECT BEST FRAME =================
        elif prev_state == 0 and state == 1:
            print("selecting best frame")

            if len(lidar_buf) < 3:
                lidar_buf.clear()
                conf_buf.clear()
                score_buf.clear()
                prev_state = state
                continue

            best_idx   = int(np.argmax(score_buf))
            lidar_best = lidar_buf[best_idx]

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


            pts_local *= total_scale
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

            print("pts_global shape:", pts_global.shape)

            # ================= 5️⃣ GLOBAL CORRECTION (yaw-aware drift suppression) =================

            pts_global_corrected = pts_global.copy()
            x_corrected, z_corrected = x, z

            # --- 센서 기준 거리 ---
            d_global = np.linalg.norm(
                pts_global - np.array([x, z], dtype=np.float32),
                axis=1
            )

            GER_mask_global = (d_global > GER_DIST_MIN) & (d_global < GER_DIST_MAX)


            # --- 현재 프레임 중심 (global) ---
            if np.count_nonzero(GER_mask_global) > 10:
                curr_center = pts_global[GER_mask_global].mean(axis=0)
            else:
                curr_center = pts_global.mean(axis=0)  # fallback


            if prev_center_local is not None and prev_yaw_refined_pose is not None:

                # === yaw 변화 ===
                dyaw = yaw_refined - prev_yaw_refined_pose
                dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

                # === 역회전 행렬 ===
                c_inv = math.cos(-dyaw)
                s_inv = math.sin(-dyaw)
                R_inv = np.array([[c_inv, s_inv],
                                  [-s_inv,  c_inv]], dtype=np.float32)

                # === 센서 기준으로 좌표 이동 후 역회전 ===
                curr_center_local = R_inv @ (curr_center - t_curr)

                # === 순수 translation drift ===
                drift_local = prev_center_local - curr_center_local

                # ---- 안전장치 ----
                DRIFT_CLAMP = 5.0
                ALPHA_CORR  = 0.5

                norm = np.linalg.norm(drift_local)
                if norm > DRIFT_CLAMP:
                    drift_local *= (DRIFT_CLAMP / norm)

                drift_local *= ALPHA_CORR

                # === 다시 현재 yaw 좌표계로 회전 ===
                c_dyaw = math.cos(dyaw)
                s_dyaw = math.sin(dyaw)
                R_dyaw = np.array([[c_dyaw, s_dyaw],
                                   [-s_dyaw,  c_dyaw]], dtype=np.float32)

                drift_global = R_dyaw @ drift_local

                # === 보정 적용 ===
                pts_global_corrected += drift_global
                x_corrected += drift_global[0]
                z_corrected += drift_global[1]

                # === 기준 갱신 ===
                prev_center_local = curr_center_local + drift_local
                prev_yaw_refined_pose = yaw_refined

            else:
                # 첫 프레임 기준 설정
                prev_center_local = curr_center - t_curr
                prev_yaw_refined_pose = yaw_refined

            delta_corrected_x = x - x_corrected
            delta_corrected_z = z - z_corrected
            
            # corrected pose 전달
            corrected_queue.append((x_corrected, z_corrected))

            #print(delta_corrected_x, delta_corrected_z)

            mx_arr = (pts_global_corrected[:, 0] / RESOLUTION) + origin
            mz_arr = (pts_global_corrected[:, 1] / RESOLUTION) + origin

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
            occ_grid *= 0.999   # 0.995~0.999 사이 추천

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
                    if angle < HFOV / 2 and scale * GER_DIST_MIN < norm_v < scale * GER_DIST_MAX:
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

            # ---------------------------------- 맵 revise local entropy based adaptive decay -----------------------------

            if map_update_count % ENTROPY_CHECK_PERIOD == 0:

                for ty in range(0, MAP_SIZE, TILE_SIZE):
                    for tx in range(0, MAP_SIZE, TILE_SIZE):

                        tile = occ_grid[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
                        if tile.size == 0:
                            continue

                        H = tile_entropy(tile)

                        # 상태 분기
                        if H > 1.6:          # CHAOTIC
                            decay = 0.90
                        elif H > 1.1:        # MID
                            decay = 0.97
                        else:                # STABLE
                            decay = 0.995

                        occ_grid[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE] *= decay

            # ---------------------------- 기하학적 slimming --------------------------------

            if map_update_count % SLIMMING_PERIOD == 0:

                for ty in range(0, MAP_SIZE, TILE_SIZE):
                    for tx in range(0, MAP_SIZE, TILE_SIZE):

                        tile = occ_grid[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]

                        ys, xs = np.where(tile > 0.2)
                        if len(xs) < 20:
                            continue

                        pts = np.stack([xs, ys], axis=1).astype(np.float32)

                        # PCA
                        mean = pts.mean(axis=0)
                        pts_c = pts - mean
                        cov = np.cov(pts_c.T)
                        eigvals, eigvecs = np.linalg.eig(cov)

                        idx = np.argmax(eigvals)
                        main_dir = eigvecs[:, idx]

                        # line-like 조건
                        if eigvals[idx] / (eigvals.sum() + 1e-6) < 0.6:
                            continue

                        # 직선 거리
                        proj = pts_c @ main_dir
                        recon = np.outer(proj, main_dir)
                        dist = np.linalg.norm(pts_c - recon, axis=1)

                        # seed (PCA에 사용한 점들)
                        seed_pts = pts

                        # tile 내 전체 점
                        all_ys, all_xs = np.where(tile > 0.2)
                        all_pts = np.stack([all_xs, all_ys], axis=1).astype(np.float32)

                        # 전체 점에 대해 line 거리 계산
                        pts_c_all = all_pts - mean
                        proj_all = pts_c_all @ main_dir
                        recon_all = np.outer(proj_all, main_dir)
                        dist_all = np.linalg.norm(pts_c_all - recon_all, axis=1)

                        # seed 보호용 set
                        seed_set = set(map(tuple, seed_pts.astype(int)))

                        # slimming: seed 제외, line에서 먼 점만 제거
                        for (x, y), d in zip(all_pts.astype(int), dist_all):
                            if (x, y) in seed_set:
                                continue
                            if d > 1.2:
                                tile[y, x] *= 0.7

            # 0~255 범위로 변환
            disp = cv2.resize((occ_grid*4*255).astype(np.uint8), (600, 600), interpolation=cv2.INTER_NEAREST)
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

            cv2.imshow("Occupancy Grid Fusion", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                 exit_flag.set()
                 break


            # ================= 8️⃣ BUFFER & HISTORY RESET =================
            global_pts_history = np.vstack([global_pts_history, pts_global_corrected])
            if len(global_pts_history) > 5000:
                global_pts_history = global_pts_history[-5000:]

            prev_state = state
            prev_pts_local = pts_local.copy()
            prev_lidar_sel = lidar_sel.copy()
            prev_lidar_norm = lidar_norm.copy()

            continue

        else:
            print("moving")
            if prev_state != state:   # 정지 → 이동 전환
                moving_corr_cnt = 0
            if prev_center_local is not None:
                dx = x - prev_x
                dz = z - prev_z

                alpha = np.clip(alpha, 0.0, 0.7)

                sup_delta_x = s_gain*alpha*dx
                sup_delta_z = s_gain*alpha*dz

                # 억제 적용
                x_supp = prev_x + sup_delta_x
                z_supp = prev_z + sup_delta_z
            else:
                x_supp, z_supp = x, z

                sup_delta_x = 0.0
                sup_delta_z = 0.0

            # suppression 결과를 기본 pose로 사용
            x_curr = x_supp
            z_curr = z_supp

            #print(sup_delta_x, sup_delta_z, alpha)
            #print(
                 #"dx:", dx,
                 #"sup_dx:", sup_delta_x,
                 #"ratio:", sup_delta_x / (dx + 1e-6)
            #)

            moving_corr_cnt += 1

            if (
                moving_corr_cnt % MOVING_CORR_EVERY == 0
                and prev_center_local is not None
                and prev_yaw_refined is not None
            ):

                # --- yaw 변화 ---
                dyaw = yaw_refined - prev_yaw_refined
                dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

                # --- 역회전 ---
                c_inv = math.cos(-dyaw)
                s_inv = math.sin(-dyaw)
                R_inv = np.array([[c_inv, s_inv],
                                  [-s_inv,  c_inv]], dtype=np.float32)

                # --- 현재 프레임 center ---
                curr_center = pts_global.mean(axis=0)
                t_curr = np.array([x_curr, z_curr], dtype=np.float32)

                curr_center_local = R_inv @ (curr_center - t_curr)

                # --- drift ---
                drift_local = prev_center_local - curr_center_local

                # --- 매우 약한 clamp ---
                norm = np.linalg.norm(drift_local)
                if norm > DRIFT_CLAMP_MOVING:
                    drift_local *= (DRIFT_CLAMP_MOVING / norm)

                drift_local *= MOVING_CORR_GAIN

                # --- 다시 global ---
                c_dyaw = math.cos(dyaw)
                s_dyaw = math.sin(dyaw)
                R_dyaw = np.array([[c_dyaw, s_dyaw],
                                   [-s_dyaw,  c_dyaw]], dtype=np.float32)

                drift_global = R_dyaw @ drift_local

                # --- pose 보정 ---
                x_curr += drift_global[0]
                z_curr += drift_global[1]

                # --- 기준 갱신 ---
                prev_center_local = curr_center_local + drift_local
                prev_yaw_refined = yaw_refined

            delta_corr_x = x_curr - x_supp
            delta_corr_z = z_curr - z_supp
            print(delta_corr_x, delta_corr_z)

            prev_x = x_curr
            prev_z = z_curr

            # corrected pose 전달
            corrected_queue.append((x_curr, z_curr))
            
            prev_state = state
            continue

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

        threading.Thread(target=receive_thread,args=(conn, raw_queue, gyro_queue),daemon=True).start()
        threading.Thread(target=imu_thread, args=( gyro_queue, delta_yaw_queue, exit_flag,),daemon=True).start()
        threading.Thread(target=depth_thread,daemon=True).start()
        threading.Thread(target=local_pose_thread, args=(processed_queue, delta_yaw_queue, motion_queue,), daemon=True).start()
        threading.Thread(target=global_pose_thread, args=(motion_queue, global_queue, exit_flag), daemon=True).start()
        threading.Thread(target=mapping_thread,args=(global_queue, corrected_queue, exit_flag), daemon=True).start()
        

        # keep main alive
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down...")

if __name__=="__main__":
    main()

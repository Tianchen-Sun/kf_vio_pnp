import numpy as np
from dataclasses import dataclass


@dataclass
class KFConfig:
    # acc noise (m/s^2)
    accel_noise_std: float = 1.0
    # vio bias random walk std (m/s per sqrt(s))
    bias_rw_std: float = 0.02

    # VIO obs noise
    vio_pos_std: float = 0.20
    vio_vel_std: float = 0.30

    # PnP obs noise (more reliable)
    pnp_pos_std: float = 0.03


class VioAugmentedKalmanFilter:
    """
    Augmented state KF
    State: [px, py, pz, vx, vy, vz, bx, by, bz]^T
      p: true position
      v: true velocity
      b: VIO position bias (estimated online)
    """

    def __init__(self, cfg: KFConfig):
        self.cfg = cfg
        self.x = np.zeros((9, 1), dtype=float)
        self.P = np.eye(9, dtype=float)
        self.t = None

    def init_state(
        self,
        pos,
        vel=(0.0, 0.0, 0.0),
        bias=(0.0, 0.0, 0.0),
        pos_var=1.0,
        vel_var=1.0,
        bias_var=0.25,  # (0.5m)^2 默认较宽松
        t0=0.0
    ):
        self.x[0:3, 0] = np.asarray(pos, dtype=float)
        self.x[3:6, 0] = np.asarray(vel, dtype=float)
        self.x[6:9, 0] = np.asarray(bias, dtype=float)

        self.P = np.diag([
            pos_var, pos_var, pos_var,
            vel_var, vel_var, vel_var,
            bias_var, bias_var, bias_var
        ]).astype(float)

        self.t = float(t0)

    def _build_F_Q(self, dt: float):
        # F: constant velocity + bias random walk
        F = np.eye(9, dtype=float)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        # Q for [p,v]
        q = self.cfg.accel_noise_std ** 2
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        Q_block = q * np.array([[dt4 / 4.0, dt3 / 2.0],
                                [dt3 / 2.0, dt2]], dtype=float)

        Q = np.zeros((9, 9), dtype=float)
        Q[np.ix_([0, 3], [0, 3])] = Q_block
        Q[np.ix_([1, 4], [1, 4])] = Q_block
        Q[np.ix_([2, 5], [2, 5])] = Q_block

        # Q for bias random walk: b_{k+1} = b_k + w_b
        qb = (self.cfg.bias_rw_std ** 2) * max(dt, 1e-6)
        Q[6, 6] = qb
        Q[7, 7] = qb
        Q[8, 8] = qb

        return F, Q

    def predict_to(self, t_now: float):
        if self.t is None:
            self.t = float(t_now)
            return

        dt = float(t_now - self.t)
        if dt <= 0.0:
            return

        F, Q = self._build_F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = float(t_now)

    def _update(self, z, H, R):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        # Joseph form
        I = np.eye(self.P.shape[0], dtype=float)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def update_vio(self, vio_pos, vio_vel):
        """
        VIO measurement model:
          z_vio = [p + b, v]
        """
        z = np.hstack([vio_pos, vio_vel])

        H = np.zeros((6, 9), dtype=float)
        # p + b
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        H[0, 6] = 1.0
        H[1, 7] = 1.0
        H[2, 8] = 1.0
        # v
        H[3, 3] = 1.0
        H[4, 4] = 1.0
        H[5, 5] = 1.0

        R = np.diag([
            self.cfg.vio_pos_std**2, self.cfg.vio_pos_std**2, self.cfg.vio_pos_std**2,
            self.cfg.vio_vel_std**2, self.cfg.vio_vel_std**2, self.cfg.vio_vel_std**2
        ]).astype(float)

        self._update(z, H, R)

    def update_pnp(self, pnp_pos):
        """
        PnP measurement model:
          z_pnp = [p]
        """
        z = np.asarray(pnp_pos, dtype=float)

        H = np.zeros((3, 9), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0

        R = np.diag([
            self.cfg.pnp_pos_std**2,
            self.cfg.pnp_pos_std**2,
            self.cfg.pnp_pos_std**2
        ]).astype(float)

        self._update(z, H, R)

    def process_event(self, event):
        """
        event:
          {
            "t": float,
            "type": "vio" or "pnp",
            "pos": [x,y,z],
            "vel": [vx,vy,vz]  # only for vio
          }
        """
        t = float(event["t"])
        self.predict_to(t)

        if event["type"] == "vio":
            self.update_vio(event["pos"], event["vel"])
        elif event["type"] == "pnp":
            self.update_pnp(event["pos"])
        else:
            raise ValueError(f"Unknown event type: {event['type']}")

    def get_state(self):
        # return (p, v, b) + full covariance
        p = self.x[0:3, 0].copy()
        v = self.x[3:6, 0].copy()
        b = self.x[6:9, 0].copy()
        return p, v, b, self.P.copy()


if __name__ == "__main__":
    cfg = KFConfig(
        accel_noise_std=1.5,
        bias_rw_std=0.01,
        vio_pos_std=0.20,
        vio_vel_std=0.30,
        pnp_pos_std=0.03
    )

    kf = VioAugmentedKalmanFilter(cfg)
    kf.init_state(
        pos=[0, 0, 0],
        vel=[0, 0, 0],
        bias=[0, 0, 0],   # 初始不确定，P里bias_var会放开
        t0=0.0
    )

    # 模拟：VIO有+0.20m x方向偏置，PnP低频但更准
    events = [
        {"t": 0.05, "type": "vio", "pos": [0.28, 0.01, 0.00], "vel": [1.5, 0.1, 0.0]},  # true约0.08 + 0.20
        {"t": 0.10, "type": "vio", "pos": [0.35, 0.02, 0.00], "vel": [1.4, 0.1, 0.0]},  # true约0.15 + 0.20
        {"t": 1.00, "type": "pnp", "pos": [1.20, 0.00, 0.00]},                            # 高可信校正
        {"t": 1.05, "type": "vio", "pos": [1.40, 0.05, 0.00], "vel": [1.3, 0.1, 0.0]},  # true约1.20 + 0.20
    ]

    import time
    for e in sorted(events, key=lambda x: x["t"]):
        start_time = time.time()
        kf.process_event(e)
        p, v, b, _ = kf.get_state()
        end_time = time.time()
        time_used = end_time - start_time
        print(f"Processed event in {time_used*1000:.2f} ms:")
        print(f"t={e['t']:.2f}, type={e['type']}, p={p}, v={v}, est_bias={b}")
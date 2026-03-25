[O下面我把你这段 **Augmented Kalman Filter（带VIO bias）代码**完整对应成数学公式，尽量做到“代码每一行 → 数学表达”。

我会分为四部分：
1️⃣ 状态定义
2️⃣ 预测模型（F, Q）
3️⃣ VIO观测模型
4️⃣ PnP观测模型 + KF更新公式

---

# 1️⃣ 状态定义（对应 `self.x`）

代码中：

```python
State: [px, py, pz, vx, vy, vz, bx, by, bz]^T
```

数学上定义为：

[
\mathbf{x}_k =
\begin{bmatrix}
\mathbf{p}_k \
\mathbf{v}_k \
\mathbf{b}_k
\end{bmatrix}
=============

\begin{bmatrix}
p_x \ p_y \ p_z \
v_x \ v_y \ v_z \
b_x \ b_y \ b_z
\end{bmatrix}
]

其中：

* (\mathbf{p})：真实位置
* (\mathbf{v})：真实速度
* (\mathbf{b})：**VIO位置偏置（关键！）**

---

# 2️⃣ 预测模型（对应 `_build_F_Q` + `predict_to`）

## 2.1 状态转移方程

代码：

```python
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt
```

对应：

[
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k , dt
]

[
\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{w}_a
]

[
\mathbf{b}_{k+1} = \mathbf{b}_k + \mathbf{w}_b
]

写成矩阵：

[
\mathbf{x}*{k+1} =
\underbrace{
\begin{bmatrix}
I & dt I & 0 \
0 & I & 0 \
0 & 0 & I
\end{bmatrix}
}*{\mathbf{F}}
\mathbf{x}_k
+
\mathbf{w}_k
]

---

## 2.2 过程噪声 Q（重点）

代码：

```python
Q_block = q * [[dt^4/4, dt^3/2],
               [dt^3/2, dt^2]]
```

这是经典：

👉 **constant acceleration model**

对应连续白噪声加速度：

[
\mathbf{w}_a \sim \mathcal{N}(0, \sigma_a^2)
]

离散后：

[
\mathbf{Q}_{pv} =
\sigma_a^2
\begin{bmatrix}
\frac{dt^4}{4} & \frac{dt^3}{2} \
\frac{dt^3}{2} & dt^2
\end{bmatrix}
]

三轴独立：

[
\mathbf{Q}*{p,v} =
\begin{bmatrix}
Q*{pv} & 0 & 0 \
0 & Q_{pv} & 0 \
0 & 0 & Q_{pv}
\end{bmatrix}
]

---

### bias random walk

代码：

```python
qb = bias_rw_std^2 * dt
```

对应：

[
\mathbf{b}_{k+1} = \mathbf{b}_k + \mathbf{w}_b,\quad
\mathbf{w}_b \sim \mathcal{N}(0, \sigma_b^2 dt)
]

---

## 2.3 协方差传播

代码：

```python
P = F P F^T + Q
```

数学：

[
\mathbf{P}_{k+1} =
\mathbf{F} \mathbf{P}_k \mathbf{F}^T + \mathbf{Q}
]

---

# 3️⃣ VIO观测模型（核心创新点）

代码：

```python
z_vio = [p + b, v]
```

---

## 3.1 数学表达

[
\mathbf{z}*{\text{vio}} =
\begin{bmatrix}
\mathbf{p} + \mathbf{b} \
\mathbf{v}
\end{bmatrix}
+
\mathbf{n}*{\text{vio}}
]

---

## 3.2 观测矩阵 H

代码：

```python
H:
p+b → 对 p 和 b 都是1
v   → 对 v 是1
```

数学：

[
\mathbf{H}_{\text{vio}} =
\begin{bmatrix}
I & 0 & I \
0 & I & 0
\end{bmatrix}
]

---

## 3.3 噪声

[
\mathbf{n}*{\text{vio}} \sim \mathcal{N}(0, \mathbf{R}*{\text{vio}})
]

[
\mathbf{R}_{\text{vio}} =
\mathrm{diag}(
\sigma_p^2, \sigma_p^2, \sigma_p^2,
\sigma_v^2, \sigma_v^2, \sigma_v^2
)
]

---

👉 **关键理解（非常重要）**

这个模型表示：

[
\text{VIO position} = \text{true position} + \text{bias}
]

所以 KF 会自动：

✅ 用 PnP 去校正 bias
✅ 用 bias 去修正 VIO

---

# 4️⃣ PnP观测模型

代码：

```python
z_pnp = p
```

---

## 数学表达

[
\mathbf{z}*{\text{pnp}} = \mathbf{p} + \mathbf{n}*{\text{pnp}}
]

---

## 观测矩阵

[
\mathbf{H}_{\text{pnp}} =
\begin{bmatrix}
I & 0 & 0
\end{bmatrix}
]

---

## 噪声

[
\mathbf{R}*{\text{pnp}} =
\sigma*{\text{pnp}}^2 I
]

---

# 5️⃣ Kalman Update（对应 `_update`）

代码：

```python
y = z - Hx
S = HPH^T + R
K = PH^T S^{-1}
x = x + Ky
```

---

## 数学

### innovation

[
\mathbf{y}_k = \mathbf{z}_k - \mathbf{H}\mathbf{x}_k
]

---

### innovation covariance

[
\mathbf{S}_k = \mathbf{H} \mathbf{P}_k \mathbf{H}^T + \mathbf{R}
]

---

### Kalman gain

[
\mathbf{K}_k = \mathbf{P}_k \mathbf{H}^T \mathbf{S}_k^{-1}
]

---

### state update

[
\mathbf{x}_k \leftarrow \mathbf{x}_k + \mathbf{K}_k \mathbf{y}_k
]

---

### covariance（Joseph form）

代码：

```python
P = (I-KH)P(I-KH)^T + KRK^T
```

数学：

[
\mathbf{P}_k =
(\mathbf{I} - \mathbf{K}\mathbf{H})
\mathbf{P}
(\mathbf{I} - \mathbf{K}\mathbf{H})^T

* \mathbf{K}\mathbf{R}\mathbf{K}^T
  ]

---

# ⭐ 总结（这段代码的本质）

这个KF做了三件非常关键的事情：

---

## ✅ 1. 建模了 VIO bias

[
\mathbf{z}_{vio} = \mathbf{p} + \mathbf{b}
]

👉 不是简单滤波，而是**系统辨识 bias**

---

## ✅ 2. 用 PnP 作为“绝对参考”

[
\mathbf{z}_{pnp} = \mathbf{p}
]

👉 类似 GPS / marker / map correction

---

## ✅ 3. 实现了 multi-rate fusion

通过：

```python
predict_to(t)
```

实现：

[
\text{asynchronous KF}
]

---

# 🚀 如果你感兴趣（很推荐）

这个模型其实就是：

👉 **Error-state EKF 的一个线性特例 + bias augmentation**

可以进一步扩展成：

* IMU + VIO + PnP（完整VIO融合）
* 加 orientation
* 加 IMU bias（gyro/acc）

---

如果你想，我可以帮你把这个直接扩展成：

👉 **完整无人机 EKF（IMU + VIO + PnP）版本（带SO(3)）**

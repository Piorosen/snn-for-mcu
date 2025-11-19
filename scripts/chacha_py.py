import numpy as np

# ---------- rate_numpy ----------
def rate_numpy(
    data,
    num_steps=None,
    gain=1.0,
    offset=0.0,
    time_var_input=False,
    first_spike_time=0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.float32)

    prob = gain * data + offset
    prob = np.clip(prob, 0.0, 1.0)

    if time_var_input:
        prob_t = prob
        T = prob_t.shape[0]
    else:
        if num_steps is None:
            raise ValueError("time_var_input=False일 때 num_steps를 지정해야 합니다.")
        T = num_steps
        prob_t = np.broadcast_to(prob, (T,) + prob.shape)

    spikes = (rng.random(prob_t.shape) < prob_t).astype(np.float32)

    if first_spike_time > 0:
        t0 = min(first_spike_time, T)
        spikes[:t0] = 0.0

    return spikes


# ---------- Conv2D ----------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.padding = padding

        limit = np.sqrt(6 / (in_channels * self.kernel_size * self.kernel_size + out_channels))
        self.weight = np.random.uniform(-limit, limit,
                                        (out_channels, in_channels, self.kernel_size, self.kernel_size)).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)

    def _pad(self, x):
        if self.padding == 0:
            return x
        return np.pad(
            x,
            ((0, 0), (0, 0),
             (self.padding, self.padding),
             (self.padding, self.padding)),
            mode="constant"
        )

    def forward(self, x):
        B, C_in, H, W = x.shape
        K = self.kernel_size
        C_out = self.out_channels

        x_p = self._pad(x)
        H_p, W_p = x_p.shape[2], x_p.shape[3]

        H_out = H_p - K + 1
        W_out = W_p - K + 1

        out = np.zeros((B, C_out, H_out, W_out), dtype=np.float32)

        for b in range(B):
            for co in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        region = x_p[b, :, h:h+K, w:w+K]
                        out[b, co, h, w] = np.sum(region * self.weight[co]) + self.bias[co]

        return out


# ---------- AvgPool2D ----------
class AvgPool2D:
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size

        H_out = H // k
        W_out = W // k

        out = np.zeros((B, C, H_out, W_out), dtype=np.float32)

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        region = x[b, c,
                                   i*k:(i+1)*k,
                                   j*k:(j+1)*k]
                        out[b, c, i, j] = np.mean(region)
        return out


# ---------- Flatten ----------
class Flatten:
    def forward(self, x):
        B = x.shape[0]
        return x.reshape(B, -1)


# ---------- Linear ----------
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def forward(self, x):
        return x @ self.weight.T + self.bias


# ---------- LeakyNP ----------
class LeakyNP:
    def __init__(
        self,
        beta=0.9,
        threshold=1.0,
        init_hidden=False,
        inhibition=False,
        reset_mechanism="subtract",
        output=True,
        graded_spikes_factor=1.0,
        reset_delay=True,
    ):
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.reset_mechanism = reset_mechanism
        self.output = output
        self.graded_spikes_factor = float(graded_spikes_factor)
        self.reset_delay = reset_delay

        self.mem = None
        self.spk = None

    def reset_state(self):
        self.mem = None
        self.spk = None

    def _apply_inhibition(self, spk, mem):
        B = spk.shape[0]
        flat_spk = spk.reshape(B, -1)
        flat_mem = mem.reshape(B, -1)
        idx_max = np.argmax(flat_mem, axis=1)
        inhibited_spk = np.zeros_like(flat_spk)
        for b in range(B):
            inhibited_spk[b, idx_max[b]] = flat_spk[b, idx_max[b]]
        return inhibited_spk.reshape(spk.shape)

    def step(self, input_, mem=None, spk_prev=None):
        x = np.asarray(input_, dtype=np.float32)

        if mem is None:
            if self.init_hidden and self.mem is not None:
                mem = self.mem
            else:
                mem = np.zeros_like(x, dtype=np.float32)
        else:
            mem = np.asarray(mem, dtype=np.float32)

        if spk_prev is None:
            if self.init_hidden and self.spk is not None:
                spk_prev = self.spk
            else:
                spk_prev = np.zeros_like(x, dtype=np.float32)
        else:
            spk_prev = np.asarray(spk_prev, dtype=np.float32)

        mem_tilde = self.beta * mem + x
        spk_raw = (mem_tilde >= self.threshold).astype(np.float32)
        spk_scaled = spk_raw * self.graded_spikes_factor

        if self.reset_mechanism == "subtract":
            mem_next = mem_tilde - spk_raw * self.threshold
        elif self.reset_mechanism == "zero":
            mem_next = mem_tilde * (1.0 - spk_raw)
        elif self.reset_mechanism == "none":
            mem_next = mem_tilde
        else:
            raise ValueError(f"Unknown reset_mechanism: {self.reset_mechanism}")

        if self.inhibition:
            spk_scaled = self._apply_inhibition(spk_scaled, mem_next)

        if self.reset_delay:
            spk_out = spk_prev
            next_spk_state = spk_scaled
        else:
            spk_out = spk_scaled
            next_spk_state = spk_scaled

        if self.init_hidden:
            self.mem = mem_next
            self.spk = next_spk_state

        return spk_out, mem_next

    def forward(self, input_, mem=None, spk_prev=None):
        return self.step(input_, mem=mem, spk_prev=spk_prev)

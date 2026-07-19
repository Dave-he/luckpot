"""
简化版LSTM彩票预测模型 (纯numpy实现)

不依赖tensorflow/pytorch, 实现一个简化版LSTM单元
- 输入: 历史window_size期的号码one-hot向量序列
- LSTM层: 64个单元 (简化版只保留输入门、遗忘门、输出门)
- 输出: 每个号码位置的概率分布
- 损失: 交叉熵

注意: 完整LSTM计算量大,这里用单层+少量epoch训练
"""
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple


class LSTMCell:
    """简化版LSTM单元 (纯numpy)"""

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rng = np.random.RandomState(seed)

        # LSTM参数: 输入门(i), 遗忘门(f), 输出门(o), 候选状态(g)
        # 拼接输入和隐藏状态
        concat_size = input_size + hidden_size
        scale = np.sqrt(2.0 / concat_size)

        # 输入门
        self.Wi = rng.randn(concat_size, hidden_size) * scale
        self.bi = np.zeros(hidden_size)
        # 遗忘门
        self.Wf = rng.randn(concat_size, hidden_size) * scale
        self.bf = np.ones(hidden_size)  # 遗忘门偏置初始化为1
        # 输出门
        self.Wo = rng.randn(concat_size, hidden_size) * scale
        self.bo = np.zeros(hidden_size)
        # 候选状态
        self.Wg = rng.randn(concat_size, hidden_size) * scale
        self.bg = np.zeros(hidden_size)

        # 缓存用于反向传播
        self.cache = []

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, X_seq):
        """前向传播, X_seq: (seq_len, input_size)"""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        self.cache = []

        for t in range(len(X_seq)):
            x = X_seq[t]
            concat = np.concatenate([x, h])

            # 输入门
            i = self._sigmoid(concat @ self.Wi + self.bi)
            # 遗忘门
            f = self._sigmoid(concat @ self.Wf + self.bf)
            # 输出门
            o = self._sigmoid(concat @ self.Wo + self.bo)
            # 候选状态
            g = np.tanh(concat @ self.Wg + self.bg)

            c = f * c + i * g
            h = o * np.tanh(c)

            self.cache.append((concat, i, f, o, g, c, h))

        return h

    def backward(self, dh_seq, lr=0.01):
        """简化反向传播 (基于时间步反向)
        dh_seq: list of dh per timestep (从后往前)
        """
        dc = np.zeros(self.hidden_size)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dWg = np.zeros_like(self.Wg)
        dbg = np.zeros_like(self.bg)

        dh = np.zeros(self.hidden_size)

        for t in reversed(range(len(self.cache))):
            concat, i, f, o, g, c, h = self.cache[t]
            dh_t = dh_seq[t] + dh

            # 输出门
            do = dh_t * np.tanh(c) * o * (1 - o)
            # 当前cell状态梯度
            dc_t = dh_t * o * (1 - np.tanh(c) ** 2) + dc
            # 遗忘门
            df = dc_t * (self.cache[t-1][5] if t > 0 else np.zeros_like(c)) * f * (1 - f)
            # 输入门
            di = dc_t * g * i * (1 - i)
            # 候选状态
            dg = dc_t * i * (1 - g ** 2)

            # 拼接梯度
            dconcat = (di @ self.Wi.T + df @ self.Wf.T + do @ self.Wo.T + dg @ self.Wg.T)
            dh = dconcat[self.input_size:]  # 传给上一个时间步的dh

            # 累积梯度
            dWi += np.outer(concat, di)
            dbi += di
            dWf += np.outer(concat, df)
            dbf += df
            dWo += np.outer(concat, do)
            dbo += do
            dWg += np.outer(concat, dg)
            dbg += dg

            dc = dc_t * f

        # 梯度裁剪
        for grad in [dWi, dWf, dWo, dWg]:
            np.clip(grad, -5, 5, out=grad)

        # 参数更新
        self.Wi -= lr * dWi
        self.bi -= lr * dbi
        self.Wf -= lr * dWf
        self.bf -= lr * dbf
        self.Wo -= lr * dWo
        self.bo -= lr * dbo
        self.Wg -= lr * dWg
        self.bg -= lr * dbg


class LSTMPredictor:
    """基于简化版LSTM的彩票号码预测器"""

    def __init__(self, config: Dict):
        self.config = config
        self.red_min, self.red_max = config["red_range"]
        self.blue_min, self.blue_max = config["blue_range"]
        self.red_count = config["red_count"]
        self.blue_count = config["blue_count"]
        self.red_total = self.red_max - self.red_min + 1
        self.blue_total = (self.blue_max - self.blue_min + 1) if self.blue_count > 0 and self.blue_max >= self.blue_min else 0

        self.seq_len = 10  # 输入序列长度
        self.hidden_size = 32  # LSTM隐藏单元数 (小一点加速)
        self.feature_size = self.red_total + max(self.blue_total, 0)
        self.epochs = 30
        self.lr = 0.05

        self.lstm = None
        # 输出层: 每个红球位置一个分类器
        self.W_out_red = None  # (hidden, red_total) for each position
        self.b_out_red = None
        self.W_out_blue = None
        self.b_out_blue = None
        self.is_trained = False

    def _to_seq_feature(self, rec: Dict) -> np.ndarray:
        """将一期开奖转为one-hot特征"""
        feat = np.zeros(self.feature_size)
        for n in rec["reds"]:
            if self.red_min <= n <= self.red_max:
                feat[n - self.red_min] = 1
        if self.blue_count > 0:
            for b in rec["blues"]:
                if self.blue_min <= b <= self.blue_max:
                    feat[self.red_total + b - self.blue_min] = 1
        return feat

    def _build_samples(self, history: List[Dict]):
        """构建训练样本: 输入seq_len期, 预测下一期"""
        X_seq, y_red, y_blue = [], [], []
        for i in range(self.seq_len, len(history)):
            seq = np.array([self._to_seq_feature(history[j])
                            for j in range(i - self.seq_len, i)])
            X_seq.append(seq)
            # 标签: 红球每个位置的类别
            y_red_pos = [r - self.red_min for r in history[i]["reds"]]
            y_red_pos = [max(0, min(self.red_total - 1, p)) for p in y_red_pos]
            while len(y_red_pos) < self.red_count:
                y_red_pos.append(0)
            y_red.append(y_red_pos[:self.red_count])
            if self.blue_count > 0:
                y_blue_pos = [b - self.blue_min for b in sorted(history[i]["blues"])]
                while len(y_blue_pos) < self.blue_count:
                    y_blue_pos.append(0)
                y_blue.append(y_blue_pos[:self.blue_count])

        return X_seq, np.array(y_red), np.array(y_blue) if y_blue else None

    def _softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def train(self, history: List[Dict]) -> Dict:
        if len(history) < self.seq_len + 10:
            return {"success": False, "error": "数据量不足"}

        X_seq, y_red, y_blue = self._build_samples(history)
        if len(X_seq) == 0:
            return {"success": False, "error": "无有效训练数据"}

        n_samples = len(X_seq)

        # 初始化LSTM和输出层
        rng = np.random.RandomState(42)
        self.lstm = LSTMCell(self.feature_size, self.hidden_size, seed=42)
        # 红球输出: 综合所有位置, 用一个共享的W_out, 输出red_total类
        self.W_out_red = rng.randn(self.hidden_size, self.red_total) * np.sqrt(2.0 / self.hidden_size)
        self.b_out_red = np.zeros(self.red_total)
        if self.blue_count > 0:
            self.W_out_blue = rng.randn(self.hidden_size, self.blue_total) * np.sqrt(2.0 / self.hidden_size)
            self.b_out_blue = np.zeros(self.blue_total)

        # 训练 (用全部样本训练, 取最后隐藏状态做预测)
        for epoch in range(self.epochs):
            total_loss = 0
            # 随机打乱
            indices = rng.permutation(n_samples)
            for idx in indices:
                seq = X_seq[idx]
                # 前向
                h = self.lstm.forward(seq)
                # 红球输出
                logits_red = h @ self.W_out_red + self.b_out_red
                probs_red = self._softmax(logits_red)

                # 损失: 多个位置的交叉熵平均
                loss = 0
                dlogits_red = np.zeros_like(probs_red)
                for pos in range(self.red_count):
                    target = y_red[idx][pos]
                    loss += -np.log(probs_red[target] + 1e-8)
                    dlogits_red[target] -= 1.0 / self.red_count
                dlogits_red += probs_red / self.red_count  # softmax梯度
                total_loss += loss / self.red_count

                # 反向传播输出层
                dW_out_red = np.outer(h, dlogits_red)
                db_out_red = dlogits_red
                dh = dlogits_red @ self.W_out_red.T

                # 蓝球
                if self.blue_count > 0 and y_blue is not None:
                    logits_blue = h @ self.W_out_blue + self.b_out_blue
                    probs_blue = self._softmax(logits_blue)
                    dlogits_blue = np.zeros_like(probs_blue)
                    for pos in range(self.blue_count):
                        target = y_blue[idx][pos]
                        loss += -np.log(probs_blue[target] + 1e-8)
                        dlogits_blue[target] -= 1.0 / max(self.blue_count, 1)
                    dlogits_blue += probs_blue / max(self.blue_count, 1)
                    dW_out_blue = np.outer(h, dlogits_blue)
                    db_out_blue = dlogits_blue
                    dh += dlogits_blue @ self.W_out_blue.T

                # LSTM反向传播 (简化: 只传最后时间步的梯度)
                dh_seq = [np.zeros(self.hidden_size) for _ in range(self.seq_len)]
                dh_seq[-1] = dh
                self.lstm.backward(dh_seq, lr=self.lr)

                # 输出层更新
                self.W_out_red -= self.lr * dW_out_red
                self.b_out_red -= self.lr * db_out_red
                if self.blue_count > 0:
                    self.W_out_blue -= self.lr * dW_out_blue
                    self.b_out_blue -= self.lr * db_out_blue

            if (epoch + 1) % 10 == 0:
                print(f"    LSTM epoch {epoch+1}/{self.epochs}, avg_loss={total_loss/n_samples:.4f}")

        self.is_trained = True
        return {"success": True, "samples": n_samples,
                "metrics": {"final_loss": round(total_loss / n_samples, 4)}}

    def predict(self, history: List[Dict]) -> Tuple[List[int], List[int], Dict]:
        if not self.is_trained or len(history) < self.seq_len:
            return [], [], {}

        # 构建最近seq_len期的序列
        seq = np.array([self._to_seq_feature(history[j])
                        for j in range(len(history) - self.seq_len, len(history))])
        h = self.lstm.forward(seq)

        # 红球预测
        logits_red = h @ self.W_out_red + self.b_out_red
        probs_red = self._softmax(logits_red)

        is_repeatable = (self.red_max - self.red_min + 1) <= 10 and self.red_count >= 3
        if is_repeatable:
            # 按位置选top1 - 但LSTM只输出一个概率分布, 这里取top red_count
            top_indices = np.argsort(probs_red)[-self.red_count:][::-1]
            reds = [int(idx) + self.red_min for idx in top_indices]
        else:
            top_indices = np.argsort(probs_red)[-self.red_count:][::-1]
            reds = sorted([int(idx) + self.red_min for idx in top_indices])

        # 蓝球
        blues = []
        blue_probs = np.zeros(max(self.blue_total, 1))
        if self.blue_count > 0:
            logits_blue = h @ self.W_out_blue + self.b_out_blue
            blue_probs = self._softmax(logits_blue)
            top_blue = np.argsort(blue_probs)[-self.blue_count:][::-1]
            blues = sorted([int(idx) + self.blue_min for idx in top_blue])

        info = {
            "red_top_probs": [(int(idx) + self.red_min, round(float(probs_red[idx]), 4))
                              for idx in np.argsort(probs_red)[-10:][::-1]],
        }
        if self.blue_count > 0:
            info["blue_top_probs"] = [(int(idx) + self.blue_min, round(float(blue_probs[idx]), 4))
                                      for idx in np.argsort(blue_probs)[-5:][::-1]]
        return reds, blues, info

    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "lstm_model.pkl"), "wb") as f:
            pickle.dump({
                "lstm_Wi": self.lstm.Wi, "lstm_bi": self.lstm.bi,
                "lstm_Wf": self.lstm.Wf, "lstm_bf": self.lstm.bf,
                "lstm_Wo": self.lstm.Wo, "lstm_bo": self.lstm.bo,
                "lstm_Wg": self.lstm.Wg, "lstm_bg": self.lstm.bg,
                "W_out_red": self.W_out_red, "b_out_red": self.b_out_red,
                "W_out_blue": self.W_out_blue, "b_out_blue": self.b_out_blue,
                "is_trained": self.is_trained,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "feature_size": self.feature_size,
            }, f)

    def load(self, model_dir: str) -> bool:
        path = os.path.join(model_dir, "lstm_model.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.seq_len = data["seq_len"]
        self.hidden_size = data["hidden_size"]
        self.feature_size = data["feature_size"]
        self.lstm = LSTMCell(self.feature_size, self.hidden_size)
        self.lstm.Wi = data["lstm_Wi"]
        self.lstm.bi = data["lstm_bi"]
        self.lstm.Wf = data["lstm_Wf"]
        self.lstm.bf = data["lstm_bf"]
        self.lstm.Wo = data["lstm_Wo"]
        self.lstm.bo = data["lstm_bo"]
        self.lstm.Wg = data["lstm_Wg"]
        self.lstm.bg = data["lstm_bg"]
        self.W_out_red = data["W_out_red"]
        self.b_out_red = data["b_out_red"]
        self.W_out_blue = data.get("W_out_blue")
        self.b_out_blue = data.get("b_out_blue")
        self.is_trained = data["is_trained"]
        return True

import numpy as np

class LinearGaussianStateSpaceModel():
    """
    線形ガウス型状態空間モデル
    """
    def __init__(self, mu0, V0, F, G, H, Q, R, y):
        """
        [状態空間モデル]
        x_t = F_t * x_t-1 + G_t * v_t
        y_t = H_t * x_t + w_t
        x_0 ~ N(mu0, V0)
        v_t ~ N(0, Q_t)
        w_t ~ N(0, R_t)

        [引数]
        mu0: ndarray(k, 1)       （次元に注意）
        V0 : ndarray(k, k)
        F:   ndarray(T, k, k)
        G:   ndarray(T, k, m)
        H:   ndarray(T, l, k)
        Q:   ndarray(T, m, m)
        R:   ndarray(T, l, l)
        y:   ndarray(T, l, 1)    （欠測値はnp.nan, 次元に注意）
        """
        # メンバ変数に保存
        self.mu0 = mu0
        self.V0 = V0
        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y

        # 各次元
        self.dim_k = F.shape[1]
        self.dim_m = G.shape[2]
        self.dim_l = H.shape[1]

        # 学習期間
        self.T = y.shape[0]

        # 次元のチェック
        self._check_dimension()

    def set_params(
        self,
        mu0=None,
        V0=None,
        F=None,
        G=None,
        H=None,
        Q=None,
        R=None,
        y=None
    ):
        """
        パラメータを設定する（更新する）
        """
        self.mu0 = mu0 if mu0 is not None else self.mu0
        self.V0 = V0 if V0 is not None else self.V0
        self.F = F if F is not None else self.F
        self.G = G if G is not None else self.G
        self.H = H if H is not None else self.H
        self.Q = Q if Q is not None else self.Q
        self.R = R if R is not None else self.R
        self.y = y if y is not None else self.y

        # 次元のチェック
        self._check_dimension()

    def kalman_filter(self, calc_liklihood=False):
        """
        カルマンフィルタを実行する
        """
        # 計算結果格納用（x~N(mu, V), y~N(nu, D)）
        mu_predicted = np.zeros((self.T, self.dim_k, 1))
        V_predicted  = np.zeros((self.T, self.dim_k, self.dim_k))
        mu_filtered = np.zeros((self.T, self.dim_k, 1))
        V_filtered  = np.zeros((self.T, self.dim_k, self.dim_k))
        nu_predicted = np.zeros((self.T, self.dim_l, 1))
        D_predicted = np.zeros((self.T, self.dim_l, self.dim_l))

        if calc_liklihood:
            ll = 0 # 対数尤度

        # カルマンフィルタを実行
        for t in range(self.T):
            # 一期先予測
            if t == 0:
                mu_predicted[0], V_predicted[0] = self._calc_x_predicted(
                    self.mu0,
                    self.V0,
                    self.F[0],
                    self.G[0],
                    self.Q[0]
                )
            else:
                mu_predicted[t], V_predicted[t] = self._calc_x_predicted(
                    mu_filtered[t-1],
                    V_filtered[t-1],
                    self.F[t],
                    self.G[t],
                    self.Q[t]
                )
            # フィルタ
            mu_filtered[t], V_filtered[t] = self._calc_x_filtered(
                mu_predicted[t],
                V_predicted[t],
                self.F[t],
                self.H[t],
                self.R[t],
                self.y[t]
            )
            # yの予測
            nu_predicted[t], D_predicted[t] = self._calc_y_predicted(
                mu_filtered[t],
                V_filtered[t],
                self.H[t],
                self.R[t]   
            )
            
            if calc_liklihood:
                ll += self._loglikelihood(
                    nu_predicted[t],
                    D_predicted[t],
                    self.y[t]
                )

        result = {
            "mu_predicted": mu_predicted,
            "V_predicted": V_predicted,
            "mu_filtered": mu_filtered,
            "V_filtered": V_filtered,
            "nu_predicted": nu_predicted,
            "D_predicted": D_predicted,
            "logliklihood": ll if calc_liklihood else None
        }
        return result
    
    def kalman_smoother(self):
        """
        固定区間平滑化を実行する
        """
        # 計算結果格納用（x~N(mu, V), y~N(nu, D)）
        mu_smoothed = np.zeros((self.T, self.dim_k, 1))
        V_smoothed  = np.zeros((self.T, self.dim_k, self.dim_k))
        nu_predicted = np.zeros((self.T, self.dim_l, 1))
        D_predicted = np.zeros((self.T, self.dim_l, self.dim_l))

        # カルマンフィルタを実行
        result_filter = self.kalman_filter()
        mu_predicted = result_filter["mu_predicted"]
        V_predicted = result_filter["V_predicted"]
        mu_filtered = result_filter["mu_filtered"]
        V_filtered = result_filter["V_filtered"]

        # 固定区間平滑化を実行
        for t in reversed(range(self.T)):
            # 固定区間平滑化
            if t == self.T-1:
                mu_smoothed[-1] = mu_filtered[-1]
                V_smoothed[-1] = V_filtered[-1]
            else:
                A = V_filtered[t] @ self.F[t+1].T @ np.linalg.inv(V_predicted[t+1])
                mu_smoothed[t] = mu_filtered[t] + A @ (mu_smoothed[t+1] - mu_predicted[t+1])
                V_smoothed[t] = V_filtered[t] + A @ (V_smoothed[t+1] - V_predicted[t+1]) @ A.T
            # yの予測
            nu_predicted[t], D_predicted[t] = self._calc_y_predicted(
                mu_smoothed[t],
                V_smoothed[t],
                self.H[t],
                self.R[t]   
            )

        result = {
            "mu_predicted": mu_predicted,
            "V_predicted": V_predicted,
            "mu_filtered": mu_filtered,
            "V_filtered": V_filtered,
            "mu_smoothed": mu_smoothed,
            "V_smoothed": V_smoothed,
            "nu_predicted": nu_predicted,
            "D_predicted": D_predicted
        }
        return result
    
    def kalman_predictor(self, F, G, H, Q, R):
        """
        長期予測を行う
        """
        # 予測時点数
        horizon = F.shape[0]

        # 計算結果格納用（x~N(mu, V), y~N(nu, D)）
        mu_predicted = np.zeros((horizon, self.dim_k, 1))
        V_predicted  = np.zeros((horizon, self.dim_k, self.dim_k))
        nu_predicted = np.zeros((horizon, self.dim_l, 1))
        D_predicted  = np.zeros((horizon, self.dim_l, self.dim_l))

        # カルマンフィルタを実行
        result = self.kalman_filter()
        mu_filtered = result["mu_filtered"]
        V_filtered = result["V_filtered"]

        # 将来予測を実行
        for t in range(horizon):
            if t == 0:
                # 1期先予測（観測値がないので予測分布=フィルタ分布）
                mu_predicted[0], V_predicted[0] = self._calc_x_predicted(
                    mu_filtered[-1],
                    V_filtered[-1],
                    F[0],
                    G[0],
                    Q[0]
                )
                # yの予測
                nu_predicted[0], D_predicted[0] = self._calc_y_predicted(
                    mu_predicted[0],
                    V_predicted[0],
                    H[0],
                    R[0]
                )
            else:
                mu_predicted[t], V_predicted[t] = self._calc_x_predicted(
                    mu_predicted[t-1],
                    V_predicted[t-1],
                    F[t],
                    G[t],
                    Q[t]
                )
                nu_predicted[t], D_predicted[t] = self._calc_y_predicted(
                    mu_predicted[t],
                    V_predicted[t],
                    H[t],
                    R[t]
                )

        result = {
            "mu_predicted": mu_predicted,
            "V_predicted": V_predicted,
            "nu_predicted": nu_predicted,
            "D_predicted": D_predicted
        }
        return result
    
    def _check_dimension(self):
        """
        与えられたパラメータの次元をチェックする
        """
        assert self.F.shape[0] == self.T, f"{self.F.shape[0]} != {self.T}"
        assert self.G.shape[0] == self.T, f"{self.G.shape[0]} != {self.T}"
        assert self.H.shape[0] == self.T, f"{self.H.shape[0]} != {self.T}"
        assert self.Q.shape[0] == self.T, f"{self.Q.shape[0]} != {self.T}"
        assert self.R.shape[0] == self.T, f"{self.R.shape[0]} != {self.T}"
        assert self.y.shape[0] == self.T, f"{self.y.shape[0]} != {self.T}"

        assert self.F.shape[1] == self.dim_k, f"{self.F.shape[1]} != {self.dim_k}"
        assert self.F.shape[2] == self.dim_k, f"{self.F.shape[2]} != {self.dim_k}"
        assert self.G.shape[1] == self.dim_k, f"{self.G.shape[1]} != {self.dim_k}"
        assert self.G.shape[2] == self.dim_m, f"{self.G.shape[2]} != {self.dim_m}"
        assert self.H.shape[1] == self.dim_l, f"{self.H.shape[1]} != {self.dim_l}"
        assert self.H.shape[2] == self.dim_k, f"{self.H.shape[2]} != {self.dim_k}"
        assert self.Q.shape[1] == self.dim_m, f"{self.Q.shape[1]} != {self.dim_m}"
        assert self.Q.shape[2] == self.dim_m, f"{self.Q.shape[2]} != {self.dim_m}"
        assert self.R.shape[1] == self.dim_l, f"{self.R.shape[1]} != {self.dim_l}"
        assert self.R.shape[2] == self.dim_l, f"{self.R.shape[2]} != {self.dim_l}"
        assert self.y.shape[1] == self.dim_l, f"{self.y.shape[1]} != {self.dim_l}"
        assert self.y.shape[2] == 1, f"{self.y.shape[2]} != {1}"

        assert self.mu0.shape[0] == self.dim_k, f"{self.mu0.shape[0]} != {self.dim_k}"
        assert self.mu0.shape[1] == 1, f"{self.mu0.shape[2]} != {1}"
        assert self.V0.shape[0] == self.dim_k, f"{self.V0.shape[0]} != {self.dim_k}"
        assert self.V0.shape[1] == self.dim_k, f"{self.V0.shape[1]} != {self.dim_k}"

    def _calc_x_predicted(self, mu0, V0, F, G, Q):
        """
        1時点先の予測分布を求める

        mu0: ndarray(k, 1)
        V0 : ndarray(k, k)
        F  : ndarray(k, k)
        G  : ndarray(k, m)
        Q  : ndarray(m, m)

        return x_predicted ~ N(mu_predicted, V_predicted)
        """
        # 1期先予測
        mu_predicted = F @ mu0
        V_predicted = F @ V0 @ F.T + G @ Q @ G.T

        return mu_predicted, V_predicted
    
    def _calc_x_filtered(self, mu_predicted, V_predicted, F, H, R, y=None):
        """
        1時点先のフィルタ分布を求める

        mu_predict: ndarray(k, 1)
        V_predict : ndarray(k, k)
        F         : ndarray(k, k)
        H         : ndarray(l, k)
        R         : ndarray(l, l)
        y         : ndarray(l, 1)

        return x_filtered ~ N(mu_filtered, V_filtered)
        """
        # フィルタ（観測値がある場合）
        if not np.isnan(y):
            # 逆行列計算の高速化は今はしない
            I = np.identity(F.shape[0])
            K = V_predicted @ H.T @ np.linalg.inv(H @ V_predicted @ H.T + R)
            mu_filtered = mu_predicted + K @ (y - H @ mu_predicted)
            V_filtered = (I - K @ H) @ V_predicted

        # フィルタ（欠測の場合）
        else:
            mu_filtered = mu_predicted
            V_filtered = V_predicted

        return mu_filtered, V_filtered

    def _calc_y_predicted(self, mu_filtered, V_filtered, H, R):
        """
        観測値の予測分布を計算する

        y = H * x + w
        x ~ N(mu, V)
        y ~ N(nu, D)
        w ~ N(0, R)

        nu : ndarray(l, 1)
        D  : ndarray(l, l)

        return y ~ N(nu, D)
        """
        nu = H @ mu_filtered
        D = H @ V_filtered @ H.T + R

        return nu, D

    def _loglikelihood(self, nu, D, y):
        """
        対数尤度を求める

        return delta_ll
        """
        if not np.isnan(y):
            delta_ll = self.dim_l * np.log(2*np.pi)
            delta_ll += np.log(np.linalg.det(D))
            delta_ll += ((y-nu).T @ np.linalg.inv(D) @ (y-nu)).item()
            delta_ll *= -0.5
        else:
            delta_ll = 0

        return delta_ll

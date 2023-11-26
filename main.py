#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import datetime

from model import LinearGaussianStateSpaceModel
from utils import *

#%%
weekday_true = [0, 100, 0, 10, 50, 100, 0]
weekday_range = [0, 20, 0, 5, 10, 20, 0]
weekday_peak_prob = [0.1, 0, 0, 0, 0, 0, 0.1]
weekday_peak_value = [20, 0, 0, 0, 0, 0, 20]
weekday_missing_prob = [0, 0.1, 0, 0, 0, 0.1, 0]
np.random.seed(0)

# weekday_true = [0, 100, 0, 10, 50, 100, 0]
# weekday_range = [0, 0, 0, 0, 0, 0, 0]
# weekday_peak_prob = [0, 0, 0, 0, 0, 0, 0]
# weekday_peak_value = [20, 0, 0, 0, 0, 0, 20]
# weekday_missing_prob = [0, 0, 0, 0, 0, 0, 0]
# np.random.seed(0)

#%%
yearly_trend = [max(int(10*np.sin(2*np.pi*i/ 365)), 0) for i in range(365)]

#%%
def make_random_data(
        weekday,
        weekday_true,
        weekday_range,
        weekday_peak_prob,
        weekday_peak_value,
        weekday_missing_prob
    ):
    
    value = weekday_true[weekday]
    value += np.random.randint(-weekday_range[weekday], weekday_range[weekday]+1)
    if np.random.rand() < weekday_peak_prob[weekday]:
        value += weekday_peak_value[weekday]
    if np.random.rand() < weekday_missing_prob[weekday]:
        value = 0

    return value

# %%
start_date = "2020-01-01"
end_date = "2021-12-31"

df_data = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
df_data["weekday"] = df_data.index.map(lambda x: x.weekday())
df_data["yt_weekly"] = df_data["weekday"].map(lambda x: weekday_true[x])
df_data["y_weekly"] = df_data["weekday"].map(
    lambda x: make_random_data(
        x,
        weekday_true,
        weekday_range,
        weekday_peak_prob,
        weekday_peak_value,
        weekday_missing_prob
    )
)

df_data["y_yearly"] = [yearly_trend[i%365] for i in range(len(df_data))]
df_data["z"] = df_data["y_yearly"].map(lambda x: int(bool(x))) * df_data["weekday"].map(lambda x: int(bool(weekday_range[x])))
df_data["y"] = df_data["y_weekly"] * df_data["y_yearly"]
df_data["y"] = df_data["y"] * df_data["z"]

#%%
df_data["y"] = df_data["y_weekly"]

#%%
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_data.index, y=df_data["y"])) 

# %% カルマンフィルタの準備
# データの長さ
T = len(df_data)

# 状態方程式の行列
F0_base_trend = np.array([[1]])
F0_pulse_trend = np.array([[1]])
F0_weekly_seasonal = np.array(
    [[-1, -1, -1, -1, -1, -1],
     [ 1,  0,  0,  0,  0,  0],
     [ 0,  1,  0,  0,  0,  0],
     [ 0,  0,  1,  0,  0,  0],
     [ 0,  0,  0,  1,  0,  0],
     [ 0,  0,  0,  0,  1,  0]]
)
F0 = make_diag_stack_matrix([F0_base_trend, F0_pulse_trend, F0_weekly_seasonal])

# システムノイズの行列
G0_base_trend = np.array([[1]])
G0_pulse_trend = np.array([[1]])
G0_weekly_seasonal = np.array(
    [[1],
     [0],
     [0],
     [0],
     [0],
     [0]]
)

G0 = make_diag_stack_matrix([G0_base_trend, G0_pulse_trend, G0_weekly_seasonal])

# 観測方程式の行列
H0_base_trend = np.array([[1]])
H0_pulse_trend = np.array([[1]]) # 暫定的に1とし後から変更
H0_weekly_seasonal = np.array([[1, 0, 0, 0, 0, 0]])
H0 = make_hstack_matrix([H0_base_trend, H0_pulse_trend, H0_weekly_seasonal, ])

# システムノイズの分散共分散行列
Q0_base_trend = np.array([[0]])
Q0_pulse_trend = np.array([[0]])
Q0_weekly_seasonal = np.array([[1e1]])

Q0 = make_diag_stack_matrix([Q0_base_trend, Q0_pulse_trend, Q0_weekly_seasonal])

# 観測ノイズの分散共分散行列（1×1行列とすることに注意）
R0 = np.array([[1e1]])

# 初期状態（データの平均・分散で適当に決定）
mu0 = np.array([[df_data["y"].mean() for i in range(F0.shape[0])]]).T
V0 = np.eye(F0.shape[0]) * df_data["y"].var()

# 観測値（T×1×1行列とすることに注意）
y = np.expand_dims(df_data["y"].values, (1, 2))

#%% 計算用にスタック
def stack_matrix(M0, N):
    """
    ndarray Mを0軸にN個重ねた ndarrayを作成する
    """
    M = np.zeros((N, M0.shape[0], M0.shape[1]))
    for n in range(N):
        M[n] = M0

    return M

F = stack_matrix(F0, T)
G = stack_matrix(G0, T)
H = stack_matrix(H0, T) # Hは時変なので後で修正
Q = stack_matrix(Q0, T)
R = stack_matrix(R0, T)

# 可視化用
H_base_trend = stack_matrix(H0_base_trend, T)
H_pulse_trend = stack_matrix(H0_pulse_trend, T)
H_weekly_seasonal = stack_matrix(H0_weekly_seasonal, T)

#%% Hの設定
def get_elapsed_days_from_newyearsday(timestamp):
    current_date = datetime.date(timestamp.year, timestamp.month, timestamp.day)
    newyearsday = datetime.date(timestamp.year, 1, 1)
    return (current_date-newyearsday).days

def make_pulse(pulse_start, pulse_end, pulse_peak, day):
    if not(0 <= pulse_start < pulse_end <= 365):
        return 0
    
    pulse_range = 0.5*(pulse_end - pulse_start)
    if pulse_start <= day <= 0.5*(pulse_start + pulse_end):
        return pulse_peak * (day-pulse_start) / pulse_range
    elif pulse_end >= day >= 0.5*(pulse_start + pulse_end):
        return pulse_peak * (pulse_end-day) / pulse_range
    else:
        return 0

# パルスのパラメータ（最尤法などで最適化する）
pulse_start = 182 # 1/1=0, 12/31=364 うるう年は無視
pulse_end = 365 # 1/1=0, 12/31=364 うるう年は無視　0 <= pulse_start < pulse_end <= 364
pulse_peak = 1 # 重みは状態変数にするため必要ない？ 

pulse = [
    make_pulse(
        pulse_start,
        pulse_end,
        pulse_peak,
        get_elapsed_days_from_newyearsday(idx)
    )
    for idx in df_data.index
]
H[:,0,1] = pulse
H_pulse_trend[:,0,0] = pulse

#%% モデル作成
model = LinearGaussianStateSpaceModel(mu0, V0, F, G, H, Q, R, y)

# %%
result_filter = model.kalman_filter(calc_liklihood=True)
result_smoother = model.kalman_smoother()

#%% 長期予測
horizon = 7*53
pred_index = pd.date_range(start=df_data.index[-1], periods=horizon+1, freq="D")[1:]
Fp = stack_matrix(F0, horizon)
Gp = stack_matrix(G0, horizon)
Hp = stack_matrix(H0, horizon)
Qp = stack_matrix(Q0, horizon)
Rp = stack_matrix(R0, horizon)

# 可視化用
Hp_base_trend = stack_matrix(H0_base_trend, horizon)
Hp_pulse_trend = stack_matrix(H0_pulse_trend, horizon)
Hp_weekly_seasonal = stack_matrix(H0_weekly_seasonal, horizon)

# H_pulseを更新
pulse = [
    make_pulse(
        pulse_start,
        pulse_end,
        pulse_peak,
        get_elapsed_days_from_newyearsday(idx)
    )
    for idx in pred_index
]
Hp[:,0,1] = pulse
Hp_pulse_trend[:,0,0] = pulse
result_predictor = model.kalman_predictor(Fp, Gp, Hp, Qp, Rp)

#%% 学習期間プロット
config = {
    "yaxis": {"min": df_data["y"].min() - 2*df_data["y"].std(), "max": df_data["y"].max() + 2*df_data["y"].std()}
}
# config = {
#     "yaxis": {"min": -10, "max": 100}
# }
base_trend = [(H_base_trend[t]*result_filter["mu_filtered"][t,0,0]).item() for t in range(T)]
pulse_trend = [(H_pulse_trend[t]*result_filter["mu_filtered"][t,1,0]).item() for t in range(T)]
weekly_seasonal = [(H_weekly_seasonal[t]@result_filter["mu_filtered"][t,2:,0]).item() for t in range(T)]
y_predicted_upper = [result_filter["nu_predicted"][t,0,0]+2*np.sqrt(result_filter["D_predicted"][t,0,0]) for t in range(T)]
y_predicted_lower = [result_filter["nu_predicted"][t,0,0]-2*np.sqrt(result_filter["D_predicted"][t,0,0]) for t in range(T)]
traces = {
    "used_data": {"x": df_data.index, "y": y[:,0,0], "color": "black", "fill": None},
    "base_trend": {"x": df_data.index, "y": base_trend, "color": "red", "fill": None},
    "pulse_trend": {"x": df_data.index, "y": pulse_trend, "color": "red", "fill": None},
    "weekly_seasonal": {"x": df_data.index, "y": weekly_seasonal, "color": "red", "fill": None},
    "y": {"x": df_data.index, "y": result_filter["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_lower": {"x": df_data.index, "y": y_predicted_lower, "color": "pink", "fill": None},
    "y_upper": {"x": df_data.index, "y": y_predicted_upper, "color": "pink", "fill": "tonexty"}
}
request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 予測プロット
base_trend = [(Hp_base_trend[t]*result_predictor["mu_predicted"][t,0,0]).item() for t in range(horizon)]
pulse_trend = [(Hp_pulse_trend[t]*result_predictor["mu_predicted"][t,1,0]).item() for t in range(horizon)]
weekly_seasonal = [(Hp_weekly_seasonal[t]@result_predictor["mu_predicted"][t,2:,0]).item() for t in range(horizon)]
y_predicted_upper = [result_predictor["nu_predicted"][t,0,0]+2*np.sqrt(result_predictor["D_predicted"][t,0,0]) for t in range(horizon)]
y_predicted_lower = [result_predictor["nu_predicted"][t,0,0]-2*np.sqrt(result_predictor["D_predicted"][t,0,0]) for t in range(horizon)]
traces = {
    "used_data": {"x": df_data.index, "y": y[:,0,0], "color": "black", "fill": None},
    "y": {"x": df_data.index, "y": result_filter["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_pred": {"x": pred_index, "y": result_predictor["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_pred_lower": {"x": pred_index, "y": y_predicted_lower, "color": "pink", "fill": None},
    "y_pred_upper": {"x": pred_index, "y": y_predicted_upper, "color": "pink", "fill": "tonexty"}
}
request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 最尤法
def objective(params):
    pulse_start = params[0] # 1/1=0, 12/31=364 うるう年は無視
    pulse_end = params[1] # 1/1=0, 12/31=364 うるう年は無視　0 <= pulse_start < pulse_end <= 364
    pulse_peak = 1 # 重みは状態変数にするため必要ない？ 

    pulse = [
        make_pulse(
            pulse_start,
            pulse_end,
            pulse_peak,
            get_elapsed_days_from_newyearsday(idx)
        )
        for idx in df_data.index
    ]
    H[:,0,1] = pulse
    H_pulse_trend[:,0,0] = pulse
    
    # パラメータ更新
    model.set_params(H=H)

    # カルマンフィルタを実行
    result = model.kalman_filter(calc_liklihood=True)

    return -1 * result["logliklihood"] # llの最大化＝-llの最小化

#%% パラメータ最適化
optimization_result = minimize(
    fun=objective,
    x0=[100, 200],
    bounds=[(0, 365), (0, 365)], 
    method="L-BFGS-B"
)

#%% 最適パラメータで計算
params = optimization_result.x

pulse_start = params[0] # 1/1=0, 12/31=364 うるう年は無視
pulse_end = params[1] # 1/1=0, 12/31=364 うるう年は無視　0 <= pulse_start < pulse_end <= 364

pulse = [
    make_pulse(
        pulse_start,
        pulse_end,
        pulse_peak,
        get_elapsed_days_from_newyearsday(idx)
    )
    for idx in df_data.index
]
H[:,0,1] = pulse
H_pulse_trend[:,0,0] = pulse

# パラメータ更新
model.set_params(H=H)

#%% カルマンフィルタを実行
result_filter = model.kalman_filter(calc_liklihood=True)
result_smoother = model.kalman_smoother()

#%% 長期予測
# 可視化用
Hp_base_trend = stack_matrix(H0_base_trend, horizon)
Hp_pulse_trend = stack_matrix(H0_pulse_trend, horizon)
Hp_weekly_seasonal = stack_matrix(H0_weekly_seasonal, horizon)

# H_pulseを更新
pulse = [
    make_pulse(
        pulse_start,
        pulse_end,
        pulse_peak,
        get_elapsed_days_from_newyearsday(idx)
    )
    for idx in pred_index
]
Hp[:,0,1] = pulse
Hp_pulse_trend[:,0,0] = pulse

result_predictor = model.kalman_predictor(Fp, Gp, Hp, Qp, Rp)

#%% 学習期間プロット
config = {
    "yaxis": {"min": df_data["y"].min() - 2*df_data["y"].std(), "max": df_data["y"].max() + 2*df_data["y"].std()}
}
# config = {
#     "yaxis": {"min": -10, "max": 100}
# }
base_trend = [(H_base_trend[t]*result_filter["mu_filtered"][t,0,0]).item() for t in range(T)]
pulse_trend = [(H_pulse_trend[t]*result_filter["mu_filtered"][t,1,0]).item() for t in range(T)]
weekly_seasonal = [(H_weekly_seasonal[t]@result_filter["mu_filtered"][t,2:,0]).item() for t in range(T)]
y_predicted_upper = [result_filter["nu_predicted"][t,0,0]+2*np.sqrt(result_filter["D_predicted"][t,0,0]) for t in range(T)]
y_predicted_lower = [result_filter["nu_predicted"][t,0,0]-2*np.sqrt(result_filter["D_predicted"][t,0,0]) for t in range(T)]
traces = {
    "used_data": {"x": df_data.index, "y": y[:,0,0], "color": "black", "fill": None},
    "base_trend": {"x": df_data.index, "y": base_trend, "color": "red", "fill": None},
    "pulse_trend": {"x": df_data.index, "y": pulse_trend, "color": "red", "fill": None},
    "weekly_seasonal": {"x": df_data.index, "y": weekly_seasonal, "color": "red", "fill": None},
    "y": {"x": df_data.index, "y": result_filter["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_lower": {"x": df_data.index, "y": y_predicted_lower, "color": "pink", "fill": None},
    "y_upper": {"x": df_data.index, "y": y_predicted_upper, "color": "pink", "fill": "tonexty"}
}
request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

#%% 予測プロット
base_trend = [(Hp_base_trend[t]*result_predictor["mu_predicted"][t,0,0]).item() for t in range(horizon)]
pulse_trend = [(Hp_pulse_trend[t]*result_predictor["mu_predicted"][t,1,0]).item() for t in range(horizon)]
weekly_seasonal = [(Hp_weekly_seasonal[t]@result_predictor["mu_predicted"][t,2:,0]).item() for t in range(horizon)]
y_predicted_upper = [result_predictor["nu_predicted"][t,0,0]+2*np.sqrt(result_predictor["D_predicted"][t,0,0]) for t in range(horizon)]
y_predicted_lower = [result_predictor["nu_predicted"][t,0,0]-2*np.sqrt(result_predictor["D_predicted"][t,0,0]) for t in range(horizon)]
traces = {
    "used_data": {"x": df_data.index, "y": y[:,0,0], "color": "black", "fill": None},
    "y": {"x": df_data.index, "y": result_filter["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_pred": {"x": pred_index, "y": result_predictor["nu_predicted"][:,0,0], "color": "red", "fill": None},
    "y_pred_lower": {"x": pred_index, "y": y_predicted_lower, "color": "pink", "fill": None},
    "y_pred_upper": {"x": pred_index, "y": y_predicted_upper, "color": "pink", "fill": "tonexty"}
}
request = {"config": config, "traces": traces}
fig = plot(request)
fig.show()

# %%

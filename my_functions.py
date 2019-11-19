import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import glob
from IPython.display import display, Audio
from scipy import stats


def euclid_divergence(Y, Yh):
    d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
    return d


def NMF(Y, n_iter, R=None, init_H=[], init_U=[], verbose=False):
    """
    Y ≈　HU
    Y ∈ R (m, n)
    H ∈ R (m, k)
    HU ∈ R (k, n)

    パラメータ
    ----
    Y: ターゲット
    R: 基底数
    n_iter: 更新回数
    init_H: Hの初期値
    init_U: Uの初期値

    return
    ----
    0: H
    1: U
    2: cost
    """

    eps = np.spacing(1)

    # スペクトログラムのサイズ
    M = Y.shape[0]
    N = Y.shape[1]

    # 初期化
    if len(init_H):
        H = init_H
        R = init_H.shape[1]
    else:
        print("ERROR!!: Please set the initial values of basis matrix")
        sys.exit(1)

    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R, N)

    # コストを保持するための入れ物
    cost = np.zeros(n_iter)

    # 近似行列
    Y_hat = np.dot(H, U)

    # イテレーション開始
    for it in range(n_iter):

        # ユークリッド距離の計算
        cost[it] = euclid_divergence(Y, Y_hat)

        # Hのアップデート
        H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)

        # Uのアップデート
        U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)

        # 近似行列の計算
        Y_hat = np.dot(H, U)

    return [H, U, cost]


def CNMF(Y, n_iter, th=0, p=1.2, a=None, init_H=[], init_U=[], verbose=False):
    """
    以下が複素NMFのモデル
    ----
    Y ≈ F = ΣHUexp(jP) = ΣHUP_exp
    Y ∈ C (K,M)　
    H ∈ R (K,L)
    U ∈ R (L,M)
    j ∈ C
    P ∈ R (L,K,M)
    B ∈ R (L,K,M)
    X ∈ R (L,K,M)

    以下が複素NMFのパラメータ
    ----
    Y:　ターゲット音源の複素スペクトログラム
    F:　モデル化された複素スペクトログラム
    L:　基底数
    n_iter:　最適化の繰り返し回数
    p：　0<p<2を満たす定数
    a：　最適化のための定数
    B:　更新パラメータ。kの行に関して和が1になる。
    init_H:　Hの初期値。行列の初期値はランダム。
    init_U:　Uの初期値。行列の初期値はランダム。
    init_P:　Pの初期値。テンソルの初期値はランダム。
    init_B:　Bの初期値。テンソルの初期値はランダム。
    X:　lに関して,ΣX = Yとなるように定められた(L,K,M)3階テンソル
    P_exp:　exp(jP)のこと。

    返り値（インデックス：　変数名）
    ----
    0:　F
    1:　H
    2:　U
    3:　P_exp
    4:　Y-F
    """
    #　行列のサイズ設定のために観測スペクトログラムのサイズを確認
    K = Y.shape[0]
    M = Y.shape[1]

    #　各変数の初期化
    if len(init_H):
        H = init_H
        K = H.shape[0]
        L = H.shape[1]
    else:
        L = 72
        H = np.random.rand(K, L)

    if len(init_U):
        U = init_U
        M = U.shape[1]
    else:
        U = np.random.rand(L, M)

    #　P_expの初期値の設定
    P0 = Y / np.abs(Y)

    #　Lの方向に次元を拡張する
    P0_ten = np.tile(P0, (L, 1))
    P_exp = P0_ten.reshape(L, K, M)

    #　Fの定義
    F_sum_hu = np.einsum("kl,lm -> lkm", H, U)
    F = np.einsum("lkm,lkm -> km", F_sum_hu, P_exp)

    #　aの生成
    if a != None:
        a = a
    else:
        a_den = np.einsum("km,km -> km", np.abs(Y), np.abs(Y))
        a_num = K**(1 - p/2)
        a = np.sum(a_den / a_num) * 10**(-5)

    #　誤差を蓄える変数の生成
    error = np.zeros(n_iter)

    #　更新を繰り返す
    for it in range(n_iter):

        #　B_norを更新
        B_num = np.einsum("kl,lm -> lkm", H, U)
        B_num_sum = np.einsum("kl,lm -> km", H, U)
        B_nor = B_num / B_num_sum

        #　Xを更新
        X_sum_hu = np.einsum("kl,lm -> lkm", H, U)
        X1 = np.einsum("lkm,lkm -> lkm", X_sum_hu, P_exp)
        X2 = np.einsum("lkm,km -> lkm", B_nor, (Y - F))
        X = X1 + X2

        #　P_expを更新
        X_abs = np.abs(X)
        P_exp = X / X_abs

        #　Uを更新
        U_den_den = np.einsum("kl,lkm -> lkm", H, X_abs)
        U_den = np.einsum("lkm,lkm -> lm", U_den_den, 1/B_nor)
        U_num_den = np.einsum("kl,kl -> kl", H, H)
        U_num = np.einsum("kl,lkm -> lm", U_num_den, 1/B_nor)
        U = U_den / (U_num + a*p*(np.abs(U))**(p-2))

        #　改めてFを生成
        F_sum_hu = np.einsum("kl,lm -> lkm", H, U)
        F = np.einsum("lkm,lkm -> km", F_sum_hu, P_exp)

        #　現在の状況の表示
        if (it/n_iter)*100 % 10 == 0:
            print("現在" + str(int((it/n_iter)*100) + 10) + "％完了")

        nm_iter = it

        # 誤差の記録
        error[it] = np.sum(np.abs(Y - F))/(K*L*M)
        if it > 20:
            if error[it] < th:
                print("閾値による強制終了")
                break

    print("繰り返し回数："+str(nm_iter+1)+"回")
    N = np.arange(1, nm_iter+2)
    error = error[:nm_iter+1]
    plt.plot(N, error)
    plt.show()

    return[error, F, H, U, P_exp, nm_iter, Y-F]


def Makespectral(wavdata, i, fs, init_H):
    K = 1025
    S = librosa.stft(y=wavdata[i, :], n_fft=2048)
    S_abs = np.abs(S)
    init_H = init_H[:, i].reshape(K, 1)
    nmf_data = NMF(Y=S_abs, R=1, n_iter=50, init_H=init_H,
                   init_U=[], verbose=False)
    return nmf_data


def initial_values(path):
    # パラメータ設定
    file_names = path+"/**/*.wav"
    wavlist_not_sorted = glob.glob(file_names, recursive=True)
    wavlist = sorted(wavlist_not_sorted)
    L = len(wavlist)
    print("The number of basis vectors："+str(L))
    K = 1025
    fs = 44100
    xlim = 11250
    l = 50
    x = np.linspace(0, xlim, xlim)
    p = np.zeros(xlim)
    pi = np.geomspace(1, 0.9**(l-1), l)

    for n in range(l):
        p += pi[n]*stats.norm.pdf(x, scale=1, loc=130.81*(n+1))
    p = p / np.sum(p)

    init_H = np.random.choice(x, K*L, p=p).reshape(K, L)

    len_list = []
    for i in range(len(wavlist)):
        len_list.append(len(librosa.load(wavlist[i])[0]))

    data_len = min(len_list)

    print("The number of frequency bins："+str(data_len))
    wavdata = np.zeros((L, data_len))

    for i in range(L):
        wavdata[i, :] = librosa.load(wavlist[i])[0][:data_len]

    wavdata = np.array(wavdata)
    spsignal = np.zeros((1025, L))

    for m in range(L):
        S_abs = Makespectral(wavdata=wavdata, i=m, fs=fs, init_H=init_H)
        spsignal[:, m] = S_abs[0].reshape(1025)

    log_power = librosa.amplitude_to_db(spsignal, ref=np.max)
    librosa.display.specshow(log_power, y_axis="log")

    plt.savefig("fixed_basis_spectrogram.png",
                format="png", bbox_inches='tight')
    return log_power

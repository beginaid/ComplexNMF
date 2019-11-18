import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from IPython.display import display, Audio


def euclid_divergence(Y, Yh):
    d = 1 / 2 * (Y ** 2 + Yh ** 2 - 2 * Y * Yh).sum()
    return d


def NMF(Y, R=3, n_iter=50, init_H=[], init_U=[], verbose=False):
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
    if len(init_U):
        U = init_U
        R = init_U.shape[0]
    else:
        U = np.random.rand(R, N)

    if len(init_H):
        H = init_H
        R = init_H.shape[1]
    else:
        H = np.random.rand(M, R)

    # コストを保持するための入れ物
    cost = np.zeros(n_iter)

    # 近似行列
    Lambda = np.dot(H, U)

    # イテレーション開始
    for i in range(n_iter):

        # ユークリッド距離の計算
        cost[i] = euclid_divergence(Y, Lambda)

        # Hのアップデート
        H *= np.dot(Y, U.T) / (np.dot(np.dot(H, U), U.T) + eps)

        # Uのアップデート
        U *= np.dot(H.T, Y) / (np.dot(np.dot(H.T, H), U) + eps)

        # 近似行列の計算
        Lambda = np.dot(H, U)

    return [H, U, cost]


def CNMF(Y, L=72, p=1.2, n_iter=10, a=None, init_H=[], init_U=[], init_P=[], init_B=[], verbose=False):
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

    #　0割を防ぐための微小定数
    eps = np.spacing(1)

    #　行列のサイズ設定のために観測スペクトログラムのサイズを確認
    K = Y.shape[0]
    M = Y.shape[1]

    #　各変数の初期化
    if len(init_H):
        H = init_H
        K = H.shape[0]
    else:
        H = np.random.rand(K, L)

    if len(init_U):
        U = init_U
        M = U.shape[1]
    else:
        U = np.random.rand(L, M)

    if len(init_P):
        P = init_P
        K = P.shape[1]
        M = P.shape[2]
    else:
        P = np.random.rand(L, K, M)

    #　P_expの初期値の設定
    P0 = Y / (np.abs(Y)+eps)

    #　Lの方向に次元を拡張する
    P0_ten = np.tile(P0, (L, 1))
    P_exp = P0_ten.reshape(L, K, M)

    #　Bだけlに関して正規化する
    if len(init_B):
        B = init_B
        K = B.shape[1]
        M = B.shape[2]
    else:
        B = np.random.rand(L, K, M)
        B_sum = np.sum(B, axis=0)
        B_nor = B / (B_sum+eps)

    #　Fの定義
    F_sum_hu = np.einsum("kl,lm -> lkm", H, U)
    F = np.einsum("lkm,lkm -> km", F_sum_hu, P_exp)

    #　aの生成
    if a != None:
        a = a
    else:
        a_den = np.einsum("km,km -> km", np.abs(Y), np.abs(Y))
        a_num = K**(1 - p/2)
        a = np.sum(a_den / (a_num+eps)) * 10**(-5)

    #　Xを生成
    X = np.zeros((L, K, M))

    #　誤差を蓄える変数の生成
    error = np.zeros(n_iter)
    N = np.arange(n_iter)

    #　更新を繰り返す
    for it in range(n_iter):

        # 誤差の記録
        error[it] = np.sum(np.abs(Y - F))

        #　Xを更新
        X_sum_hu = np.einsum("kl,lm -> lkm", H, U)
        X1 = np.einsum("lkm,lkm -> lkm", X_sum_hu, P_exp)
        X2 = np.einsum("lkm,km -> lkm", B_nor, (Y - F))
        X = X1 + X2

        #　P_expを更新
        X_abs = np.abs(X)
        P_exp = X / (X_abs+eps)

        #　Hを更新
        H_den_den = np.einsum("lm,lkm -> lkm", U, X_abs)
        H_den = np.einsum("lkm,lkm -> kl", H_den_den, 1/B_nor)
        H_num_den = np.einsum("lm,lm -> lm", U, U)
        H_num = np.einsum("lm,lkm -> kl", H_num_den, 1/B_nor)
        H = H_den / (H_num+eps)

        #　Hをkに関して正規化
        H_sum = np.sum(H, axis=0)
        H = H / (H_sum+eps)

        #　Uを更新
        U_den_den = np.einsum("kl,lkm -> lkm", H, X_abs)
        U_den = np.einsum("lkm,lkm -> lm", U_den_den, 1/B_nor)
        U_num_den = np.einsum("kl,kl -> kl", H, H)
        U_num = np.einsum("lkm,lkm -> lm", U_den_den, 1/B_nor)
        U = U_den / (U_num + a*p*(np.abs(U))**(p-2) + eps)

        #　B_norを更新
        B_num = np.einsum("kl,lm -> lkm", H, U)
        B_num_sum = np.einsum("kl,lm -> km", H, U)
        B_nor = B_num / (B_num_sum + eps)

        #　改めてFを生成
        F_sum_hu = np.einsum("kl,lm -> lkm", H, U)
        F = np.einsum("lkm,lkm -> km", F_sum_hu, P_exp)

    plt.plot(N, error)
    plt.show()

    return [a, error[n_iter - 1], F, H, U, P_exp, Y - F]


def SCNMF(Y, L=72, p=1.2, n_iter=10, th=0, a=None, init_H=[], init_U=[], verbose=False):
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
        U_num = np.einsum("lkm,lkm -> lm", U_den_den, 1/B_nor)
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

        if it > 0:
            if error[it] > error[it - 1]:
                max_error = error[it]
            if (error[it] / max_error) < 0.05611:
                print("誤差の割合による強制終了")
                break

    print("繰り返し回数："+str(nm_iter+1)+"回")
    N = np.arange(1, nm_iter+2)
    error = error[:nm_iter+1]
    plt.plot(N, error)
    plt.show()

    return[a, error[nm_iter], F, H, U, P_exp, nm_iter, Y-F]


def initial_values():
    nm_cr = 12*3
    nm_pf = 12*3
    L = nm_cr + nm_pf
    K = 1025
    fs = 44100
    xlim = 11250
    l = 50
    x = np.linspace(0, xlim, xlim)
    f = np.linspace(0, l, l)
    p_cr = np.zeros(xlim)
    p_pf = np.zeros(xlim)
    pi_cr = np.geomspace(1, 0.9**(l-1), l)
    pi_pf = np.geomspace(1, 0.9**(l-1), l)

    for n in range(l):
        p_cr += pi_cr[n]*stats.norm.pdf(x, scale=1, loc=261.63*(n+1))
    p_cr = p_cr / np.sum(p_cr)

    for m in range(l):
        p_pf += pi_pf[m]*stats.norm.pdf(x, scale=1, loc=130.81*(m+1))
    p_pf = p_pf / np.sum(p_pf)

    init_H_cr = random.choice(x, K*nm_cr, p=p_cr).reshape(K, nm_cr)
    init_H_pf = random.choice(x, K*nm_pf, p=p_pf).reshape(K, nm_pf)
    init_H = np.concatenate((init_H_cr, init_H_pf), axis=1)

    sound = AudioSegment.from_file("0_bs_div0-0.wav", "wav")

    # 情報の取得
    time = sound.duration_seconds  # 再生時間(秒)
    rate = sound.frame_rate  # サンプリングレート(Hz)
    channel = sound.channels  # チャンネル数(1:mono, 2:stereo)

    # 情報の表示
    print('チャンネル数：', channel)
    print('サンプリングレート：', rate)
    print('再生時間：', time)

    data_len = len(librosa.load("cr" + str(0)+".wav")[0])
    wavdata_cr = np.zeros((nm_cr, data_len))
    wavdata_pf = np.zeros((nm_pf, data_len))

    for i in range(nm_cr):
        wavdata_cr[i, :] = librosa.load("cr" + str(i)+".wav")[0]

    for j in range(nm_pf):
        wavdata_pf[j, :] = librosa.load("pf" + str(j)+".wav")[0]

    wavdata_cr = np.array(wavdata_cr)
    wavdata_pf = np.array(wavdata_pf)

    spsignal_cr = np.zeros((1025, nm_cr))
    for m in range(nm_cr):
        S_abs = Makespectral(wavdata=wavdata_cr, i=m, fs=fs, init_H=init_H)
        spsignal_cr[:, m] = S_abs[0].reshape(1025)

    spsignal_pf = np.zeros((1025, nm_pf))
    for m in range(nm_pf):
        S_abs = Makespectral(wavdata=wavdata_pf, i=m, fs=fs, init_H=init_H)
        spsignal_pf[:, m] = S_abs[0].reshape(1025)
    spsignal = np.concatenate((spsignal_cr, spsignal_pf), axis=1)

    # load wav
    y_crpf, sr_crpf = librosa.load(str(nm_musics)+'.wav')

    plt.subplot(311)
    plt.title('mixed')
    plt.plot(y_crpf)

    print('ソース音源: MIX音')
    display(Audio(y_crpf, rate=sr_crpf))
    print("サンプリング周波数：" + str(rate))

    # STFT
    S_crpf = librosa.stft(y=y_crpf, n_fft=2048)

    # 学習で行う反復計算回数
    n_iter = 100

    nmf = NMF(Y=np.abs(S_crpf), R=L, n_iter=n_iter, init_H=spsignal)
    init_U = nmf[1]
    np.save(str(nm_musics)+"init_U", init_U)
    K = S_crpf.shape[0]
    L = nm_cr + nm_pf
    M = S_crpf.shape[1]

    # 縦軸を対数表示
    fig = plt.figure(1, figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    log_power = librosa.amplitude_to_db(spsignal, ref=np.max)
    librosa.display.specshow(log_power, y_axis="log")
    plt.savefig('教師基底行列のスペクトログラム.svg', format="svg", bbox_inches='tight')

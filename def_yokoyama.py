
# %%
import numpy as np
import sporco.prox as pr
import sporco.cnvrep as cnvrep
from sporco import util
from sporco import plot
import matplotlib.pyplot as plt
import sporco.metric as sm
from skimage.metrics import structural_similarity as ssim

from typing import TypedDict


# %%
# Aのリゾルベントの右部分の計算
def make_migi(IDDt, df, byf, N, M, alpha):
    Dtb = np.zeros(N * M, dtype=np.complex)
    migi_NM = np.zeros(N * M)
    migi_N = np.fft.ifft(byf / IDDt)
    migi_N = migi_N.astype(np.float64)
    for i in range(M):
        Dtb[i * N:(i + 1) * N] = df[i * N:(i + 1) * N].conjugate() / IDDt * byf * alpha * -1
    for i in range(M):
        migi_NM[i * N:(i + 1) * N] = np.fft.ifft(Dtb[i * N:(i + 1) * N])
    migi_NM = migi_NM.astype(np.float64)
    return np.concatenate([migi_NM, migi_N], 0)


# Aのリゾルベントの左部分の計算
def make_hidari(IDDt, df, bxf, N, M, alpha):
    DtDb = np.zeros(N * M, dtype=np.complex)
    Db = np.zeros(N, dtype=np.complex)
    for i in range(M):
        Db = df[i * N:(i + 1) * N] * bxf[i * N:(i + 1) * N] + Db
    hidari_N = alpha * Db / IDDt
    for i in range(M):
        DtDb[i * N:(i + 1) * N] = df[i * N:(i + 1) * N].conjugate() * hidari_N
    DtDb = bxf - (alpha) * DtDb
    hidari_N = np.fft.ifft(hidari_N)
    hidari_NM = np.zeros(N * M)
    for i in range(M):
        hidari_NM[i * N:(i + 1) * N] = np.fft.ifft(DtDb[i * N:(i + 1) * N])
    hidari_N = hidari_N.astype(np.float64)
    hidari_NM = hidari_NM.astype(np.float64)
    return np.concatenate([hidari_NM, hidari_N], 0)

def make_y(df, x, N, M):
    #  h(Dx)=l1norm.(Dx - s)  y = Dxを初期値とした
    xf = np.zeros(x.shape, dtype=np.complex)
    for i in range(M):
        xf[i * N:(i + 1) * N] = np.fft.fft(x[i * N:(i + 1) * N])
    yf = np.zeros(N, dtype=np.complex)
    for i in range(M):
        yf = df[i * N:(i + 1) * N] * xf[i * N:(i + 1) * N] + yf
    y = np.fft.ifft(yf)
    y = y.astype(np.float64)
    return y


def D_to_dd_conj(D, dy, xf, N, d_size):
    flat = np.zeros(N * D.shape[0])
    for i in range(D.shape[0]):
        a = zero_pad(D[i], N, d_size)
        flat[i * N:(i + 1) * N] = np.ravel(a)
    # emptyの方がメモリ抑えられるかも
    # df = np.zeros(N * N * D.shape[0], dtype=np.complex)
    # for i in range(D.shape[0]):
    #     df[i * N:(i + 1) * N] = np.fft.fft(flat[i * N:(i + 1) * N])
    # yf = np.zeros(N, dtype=np.complex)
    # for i in range(D.shape[0]):
    #     yf = df[i * N:(i + 1) * N] * xf[i * N:(i + 1) * N] + yf
    # y = np.fft.ifft(yf) * np.sqrt(N)
    # y = y.astype(np.float64)
    return np.concatenate([flat, dy], 0)

def resolvent_d(d, y, s, rho, N, M, d_size):
    rootN = int(np.sqrt(N))
    d = d.reshape(M, rootN, rootN)
    d = d.transpose(1, 2, 0)
    d_Pcn = cnvrep.Pcn(d.reshape(rootN, rootN, 1, 1, M),
                       (d_size, d_size, M), Nv=(rootN, rootN)).squeeze()
    d_Pcn = d_Pcn.transpose(2, 0, 1)
    d = d_Pcn.reshape(N * M)
    y = y - (pr.prox_l1(y - s, 1 / rho) + s)
    return np.concatenate([d, y], 0)


def pr_d(d, N, M, d_size):
    rootN = int(np.sqrt(N))
    d = d.reshape(M, rootN, rootN)
    d = d.transpose(1, 2, 0)
    d_Pcn = cnvrep.Pcn(d.reshape(rootN, rootN, 1, 1, M),
                       (d_size, d_size, M), Nv=(rootN, rootN)).squeeze()
    d = d_Pcn.transpose(2, 0, 1)
    return d


def dictinary_learning(D, dcon, xf, s, N, M, rho, d_size, ite=50):
    dd_conj = []
    prlog = []
    dulog = []
    count = 0
    for i in range(len(xf)):
        dd_conj.append(D_to_dd_conj(D, dcon, xf, N, d_size))
    while True:
        pmove = 0
        dmove = 0
        for j in range(len(dd_conj)):
            re_d = resolvent_d(
                dd_conj[j][:N * M], dd_conj[j][N * M:], s[j], rho, N, M, d_size)
            b = 2 * re_d - dd_conj[j]
            bf = np.zeros(b.shape, dtype=np.complex)
            for k in range(M + 1):
                bf[k * N:(k + 1) * N] = np.fft.fft(b[k * N:(k + 1) * N])
            IXXt = make_IDDt(xf[j], N, M, rho)
            hidari = make_hidari(IXXt, xf[j], bf[:N * M], N, M, rho)
            migi = make_migi(IXXt, xf[j], bf[N * M:], N, M, rho)
            dd_conj[j] = dd_conj[j] + migi + hidari - re_d
            # rsdl_n = max(np.linalg.norm(migi[:N * M] + hidari[:N * M]), np.linalg.norm(re_d[:N * M]))
            # rsdl_nd = max(np.linalg.norm(migi[N * M:] + hidari[N * M:]), np.linalg.norm(re_d[N * M:]))
            # pmove = np.linalg.norm(migi[:N * M] + hidari[:N * M] - re_d[:N * M]) / rsdl_n + pmove
            # dmove = np.linalg.norm(migi[N * M:] + hidari[N * M:] - re_d[N * M:]) / rsdl_nd + dmove
            # move = np.linalg.norm(migi + hidari - re_d) / rsdl_n + move
            # print("move[" + str(j + 1) + "] : ", np.linalg.norm(migi + hidari - re_d) / rsdl_n)
            pmove = np.linalg.norm(dd_conj[j][:N * M] - re_d[:N * M]) + pmove
            dmove = np.linalg.norm(dd_conj[j][N * M:] - re_d[N * M:]) + dmove
        prlog.append(pmove / len(dd_conj))
        dulog.append(dmove / len(dd_conj))
        d_mean = sum(dd_conj) / len(dd_conj)
        # d = pr_d(d_mean[:N * M], N, M, d_size)
        for kk in range(len(dd_conj)):
            dd_conj[kk][:N * M] = d_mean[:N * M]
        # gosa, hizero = check(xf, dd_conj, s, N, M, lamd)
        count = count + 1
        print("count :", count)
        if count >= ite:
            break
    print("final dic count :", count)
    d = pr_d(d_mean[:N * M], N, M, d_size)
    return d, d_mean[N * M:], prlog, dulog


def dx_s(D, xf, s, N, M, K, d_size):
    samu = 0.
    df = D_to_df(D, N, d_size)
    for k in range(K):
        dxf = np.zeros(N, dtype=np.complex)
        for i in range(M):
            dxf = df[i * N:(i + 1) * N] * xf[k][i * N:(i + 1) * N] + dxf
        dx = np.fft.ifft(dxf)
        dx = dx.astype(np.float64)
        dx_s = dx - s[k]
        samu = np.linalg.norm(dx_s, ord=1) + samu
    return samu


def l1x(X, N, M):
    samu = 0
    for i in range(len(X)):
        for j in range(M):
            samu = np.linalg.norm(X[i][j * N:(j + 1) * N], ord=1) + samu
    return samu


def dictinary_learning_l2(D, dcon, xf, s, N, M, rho, d_size, ite=50):
    dd_conj = []
    prlog = []
    dulog = []
    count = 0
    for i in range(len(xf)):
        dd_conj.append(D_to_dd_conj(D, dcon, xf, N, d_size))
    while True:
        pmove = 0
        dmove = 0
        for j in range(len(dd_conj)):
            re_d = resolvent_d_l2(
                dd_conj[j][:N * M], dd_conj[j][N * M:], s[j], rho, N, M, d_size)
            b = 2 * re_d - dd_conj[j]
            bf = np.zeros(b.shape, dtype=np.complex)
            for k in range(M + 1):
                bf[k * N:(k + 1) * N] = np.fft.fft(b[k * N:(k + 1) * N])
            IXXt = make_IDDt(xf[j], N, M, rho)
            hidari = make_hidari(IXXt, xf[j], bf[:N * M], N, M, rho)
            migi = make_migi(IXXt, xf[j], bf[N * M:], N, M, rho)
            dd_conj[j] = dd_conj[j] + migi + hidari - re_d
            # rsdl_n = max(np.linalg.norm(migi[:N * M] + hidari[:N * M]), np.linalg.norm(re_d[:N * M]))
            # rsdl_nd = max(np.linalg.norm(migi[N * M:] + hidari[N * M:]), np.linalg.norm(re_d[N * M:]))
            # pmove = np.linalg.norm(migi[:N * M] + hidari[:N * M] - re_d[:N * M]) / rsdl_n + pmove
            # dmove = np.linalg.norm(migi[N * M:] + hidari[N * M:] - re_d[N * M:]) / rsdl_nd + dmove
            # move = np.linalg.norm(migi + hidari - re_d) / rsdl_n + move
            # print("move[" + str(j + 1) + "] : ", np.linalg.norm(migi + hidari - re_d) / rsdl_n)
            pmove = np.linalg.norm(dd_conj[j][:N * M] - re_d[:N * M]) + pmove
            dmove = np.linalg.norm(dd_conj[j][N * M:] - re_d[N * M:]) + dmove
        prlog.append(pmove / len(dd_conj))
        dulog.append(dmove / len(dd_conj))
        d_mean = sum(dd_conj) / len(dd_conj)
        # d = pr_d(d_mean[:N * M], N, M, d_size)
        for kk in range(len(dd_conj)):
            dd_conj[kk][:N * M] = d_mean[:N * M]
        # gosa, hizero = check(xf, dd_conj, s, N, M, lamd)
        count = count + 1
        print("count :", count)
        if count >= ite:
            break
    print("final dic count :", count)
    d = pr_d(d_mean[:N * M], N, M, d_size)
    return d, d_mean[N * M:], prlog, dulog


def resolvent_d_l2(d, y, s, rho, N, M, d_size):
    rootN = int(np.sqrt(N))
    d = d.reshape(M, rootN, rootN)
    d = d.transpose(1, 2, 0)
    d_Pcn = cnvrep.Pcn(d.reshape(rootN, rootN, 1, 1, M),
                       (d_size, d_size, M), Nv=(rootN, rootN)).squeeze()
    d_Pcn = d_Pcn.transpose(2, 0, 1)
    d = d_Pcn.reshape(N * M)
    y = y - (pr.prox_l2(y - s, 1 / rho) + s)
    return np.concatenate([d, y], 0)


class StatLog:
    def __init__(self):
        self.reset()

    def log(self, s: str, val: float):
        if s == "l0":
            self.l0.append(val)
        elif s == "primal_x":
            self.primal_x.append(val)
        elif s == "primal_d":
            self.primal_d.append(val)
        elif s == "dual_x":
            self.dual_x.append(val)
        elif s == "dual_d":
            self.dual_d.append(val)

    def reset(self):
        self.primal_x = []
        self.primal_d = []
        self.dual_x = []
        self.dual_d = []
        self.l0 = []


class DR_DctLnL1:
    class Options(TypedDict):
        lmbda: float
        rhox: float
        rhod: float
        coef_ite: int
        dict_ite: int
        total_ite: int

    def __init__(self, D0, S, opt: Options):
        self.opt = opt
        self.K = S.shape[0]
        self.M = D0.shape[0]
        self.N = np.prod(S.shape[1:])
        self.s_size = S.shape[1:]

        self.axisK = 0
        self.axisM = 1
        self.axisN = 2

        self.lmbda = self.opt["lmbda"]
        self.rhox = self.opt["rhox"]
        self.rhod = self.opt["rhod"]


        self.S = S.reshape(self.K, 1, self.N)
        self.Sf = np.fft.rfft(self.S)

        self.dinit(D0)
        self.xinit()

        self.stats = StatLog()
        self.j = 0

    def pcn(self, d):
        d = d.reshape(self.M, *self.s_size)
        d[:, :, self.d_size[1]:] = 0
        d[:, self.d_size[0]:, :self.d_size[1]] = 0
        d = d.reshape(1, self.M, self.N)
        tmp = np.linalg.norm(d, axis=self.axisN)
        np.putmask(tmp, tmp > 1, 1)
        return d / tmp.reshape(1, self.M, 1)

    def dinit(self, D0):
        self.d_size = D0.shape[1:]
        self.ZD = np.pad(D0, ((0, 0), (0, S.shape[1] - D0.shape[1]), (0, S.shape[2] - D0.shape[2]))).reshape(1, self.M, self.N)
        self.D = self.pcn(self.ZD)
        self.ZE = np.random.randn(self.K, 1, self.N)
        self.E = self.ZE - self.rhod * (pr.prox_l1(self.ZE / self.rhod - self.S, 1 / self.rhox) + self.S)
        self.Df = np.fft.rfft(self.D)

    def xinit(self):
        self.X = np.zeros((self.K, self.M, self.N), dtype=float)
        self.ZY = np.random.randn(self.K, 1, self.N)
        self.ZX = self.X.copy()
        self.Y = self.ZY - self.rhox * (pr.prox_l1(self.ZY / self.rhox - self.S, 1 / self.rhox) + self.S)

    def dstep(self):
        Xf = self.Xf
        rho = self.rhod
        IXXtf = 1 * rho * rho * np.sum(np.moveaxis(Xf, 1, 0) * Xf.reshape(self.K, self.M, 1, -1), axis=1)
        for _ in range(self.opt["dict_ite"]):
            Raf4E = np.sum(np.fft.rfft(2 * self.E - self.ZE).reshape(1, self.K, -1) / IXXtf, axis=1, keepdims=True)
            Raf2E = -rho * np.sum(np.conj(Xf) * Raf4E, axis=0, keepdims=True)
            bD = np.fft.rfft(2 * self.D - self.ZD)
            Raf3D = rho * np.sum(np.sum(Xf * bD, axis=1, keepdims=True) / IXXtf, axis=1, keepdims=True)
            Raf1D = bD - rho * np.sum(np.conj(Xf) * Raf3D, axis=0)
            self.ZD += np.fft.irfft(Raf1D + Raf2E) - self.D
            self.ZE += np.fft.irfft(Raf3D + Raf4E) - self.E
            self.D = self.pcn(self.ZD)
            self.E = self.ZE - rho * (pr.prox_l1(self.ZE / rho - self.S, 1 / rho) + self.S)
        self.Df = np.fft.rfft(self.D)

    def xstep(self):
        Df = self.Df
        rho = self.rhox
        lmbda = self.lmbda
        IDDtf = 1 + rho * rho * np.sum(Df * np.conj(Df), axis=self.axisM)
        for _ in range(self.opt["coef_ite"]):
            Raf4Y = np.fft.rfft(2 * self.Y - self.ZY) / IDDtf
            Raf2Y = -rho * np.conj(Df) * Raf4Y
            bX = np.fft.rfft(2 * self.X - self.ZX)
            Raf3X = rho * np.sum(Df * bX, axis=1, keepdims=True) / IDDtf
            Raf1X = bX - rho * np.conj(Df) * Raf3X
            self.ZX += np.fft.irfft(Raf1X + Raf2Y) - self.X
            self.ZY += np.fft.irfft(Raf3X + Raf4Y) - self.Y
            self.X = pr.prox_l1(self.ZX, lmbda * rho)
            self.Y = self.ZY - rho * (pr.prox_l1(self.ZY / rho - self.S, 1 / rho) + self.S)
        self.Xf = np.fft.rfft(self.X)

    def step(self):
        self.xstep()
        self.dstep()

    def solve(self):
        for self.j in range(self.j, self.j + self.opt["total_ite"]):
            self.step()

    def reconstruct(self):
        if not hasattr(self, "Xf"):
            return np.fft.irfft(np.sum(self.Df * np.fft.rfft(self.X), axis=self.axisM)).reshape(self.K, *self.s_size)
        return np.fft.irfft(np.sum(self.Df * self.Xf, axis=self.axisM)).reshape(self.K, *self.s_size)
    
    def getdict(self):
        return self.D.reshape(self.M, *self.s_size)[:, :self.d_size[0], :self.d_size[1]]


# %%
zoom = 0.25
N = int(512 * 0.25)
K = 5

exim = util.ExampleImages(scaled=True, zoom=zoom, gray=True)
s1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
s2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
s3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
s4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
s5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
ori = np.asarray([s1, s2, s3, s4, s5])

npd = 16
fltlmbd = 10

sl, sh = util.tikhonov_filter(np.moveaxis(ori, 0, -1), fltlmbd, npd)
sl = np.moveaxis(sl, -1, 0)
sh = np.moveaxis(sh, -1, 0)

S = sh.reshape(5, N, N)

# 辞書枚数
M = 16
d_size = 10
K = len(S)

D0 = np.random.randn(M, d_size, d_size)


roopcount = 0

# %%
lamd = 0.03
rhox = 1
rhod = 1
total_ite = 50
opt = DR_DctLnL1.Options(lmbda=lamd, rhox=rhox, rhod=rhod, total_ite=total_ite, coef_ite=10, dict_ite=1)
d = DR_DctLnL1(D0, (S[0:1] - 0.5) * 128, opt)
d.solve()
plot.imview(util.tiledict(d.reconstruct().transpose(1, 2, 0)))
plot.imview(util.tiledict(d.getdict().transpose(1, 2, 0)))
print(np.sum(d.X != 0, axis=(d.axisM, d.axisN)))
# print(sm.psnr(S[0], d.reconstruct()[0]))
# %%
# L1 - L1 DR
for i in range(total_ite):
    roopcount = roopcount + 1
    X, y, prx, dux, hize = coefficient_learning(d, X, y, S, N * N, M, rhox, lamd, d_size, coef_ite)
    # logging
    prx_log.append(prx[coef_ite - 1])
    dux_log.append(dux[coef_ite - 1])
    hizero.append(hize)
    # fft
    xf = X_to_xf(X, N * N, M)
    d, dcon, prd, dud = dictinary_learning(d, dcon, xf, S, N * N, M, rhod, d_size, dict_ite)
    # logging
    prd_log.append(prd[0])
    dud_log.append(dud[0])
    # prox of support function
    d = pr_d(d, N * N, M, d_size)
    # X = X.transpose(1, 2, 0)
    # crop
    d = d[:, :d_size, :d_size]
    if i == total_ite - 1:
        d = d.transpose(1, 2, 0)
        plot.imview(util.tiledict(d), fgsz=(7, 7))
        d = d.transpose(2, 0, 1)
    mindx_s = dx_s(d, xf, S, N * N, M, K, d_size) + lamd * l1x(X, N * N, M)
    goal.append(mindx_s)
    D = D_to_d(d, N * N, d_size)
    image = reconstruct(xf[0], D, N * N, M)
    print("iterate number: ", roopcount)
    imgr = sl1 + image
    # imgr = image
    print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(ori[0], imgr))
# - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
# L2 - L1 DR
# - -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 

# for i in range(total_ite):
#     roopcount = roopcount + 1
#     X, y, prx, dux, hize = coefficient_learning_l2(
#         d, X, y, S, N * N, M, rhox, lamd, d_size, coef_ite)
#     prx_log.append(prx[coef_ite - 1])
#     dux_log.append(dux[coef_ite - 1])
#     hizero.append(hize)
#     xf = X_to_xf(X, N * N, M)
#     d, dcon, prd, dud = dictinary_learning_l2(
#         d, dcon, xf, S, N * N, M, rhod, d_size, dict_ite)
#     prd_log.append(prd[0])
#     dud_log.append(dud[0])
#     d = pr_d(d, N * N, M, d_size)
#     # X = X.transpose(1, 2, 0)
#     d = d[:, :d_size, :d_size]
#     if i == total_ite - 1:
#         d = d.transpose(1, 2, 0)
#         plot.imview(util.tiledict(d), fgsz=(7, 7))
#         d = d.transpose(2, 0, 1)
#     mindx_s = dx_s(d, xf, S, N * N, M, K, d_size) + lamd * l1x(X, N * N, M)
#     goal.append(mindx_s)
#     D = D_to_d(d, N * N, d_size)
#     image = reconstruct(xf[0], D, N * N, M)
#     print("iterate number: ", roopcount)
#     imgr = sl1 + image
#     # imgr = image
#     print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(ori[0], imgr))
#     # fig = plot.figure(figsize=(14, 7))
#     # plot.subplot(1, 2, 1)
#     # plot.imview(ori[0], title='Original', fig=fig)
#     # plot.subplot(1, 2, 2)
#     # plot.imview(image / N * 10 + sl1, title='Reconstructed[' + str(i + 1) + "]" + str(sm.psnr(ori[0], imgr)), fig=fig)
#     # for j in range(len(xf)):
#     #     image = reconstruct(xf[j], D, N * N, M)
#     #     fig = plot.figure(figsize=(14, 7))
#     #     plot.subplot(1, 2, 1)
#     #     plot.imview(ori[j], title='Original', fig=fig)
#     #     plot.subplot(1, 2, 2)
#     #     plot.imview(image, title='Reconstructed[' + str(i + 1) + "]", fig=fig)
# %%
print(sh1[0][0] / image[0][0])
print(sh1)
print(image)
print(np.array(X).shape)
# %%
imgr = []
imgr.append(sl1 + reconstruct(xf[0], D, N * N, M))
imgr.append(sl2 + reconstruct(xf[1], D, N * N, M))
imgr.append(sl3 + reconstruct(xf[2], D, N * N, M))
imgr.append(sl4 + reconstruct(xf[3], D, N * N, M))
imgr.append(sl5 + reconstruct(xf[4], D, N * N, M))
# imgr.append(reconstruct(xf[0], D, N * N, M))
# imgr.append(reconstruct(xf[1], D, N * N, M))
# imgr.append(reconstruct(xf[2], D, N * N, M))
# imgr.append(reconstruct(xf[3], D, N * N, M))
# imgr.append(reconstruct(xf[4], D, N * N, M))
for g in range(K):
    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(ori[g], title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(imgr[g], title='Reconstructed[' + str(i + 1) + "]" +
                str(sm.psnr(ori[g], imgr[g])), fig=fig)
    print("ssim", ssim(ori[g], imgr[g]))
    # print(measure.compare_ssim(imgr[g], ori[g]))
# fig = plot.figure(figsize=(14, 7))
# plot.subplot(1, 2, 1)
# plot.imview(sh1, title='Original', fig=fig)
# plot.subplot(1, 2, 2)
# plot.imview(imgr[0], title='Reconstructed[' + str(i + 1) + "]" + str(sm.psnr(sh1, imgr[0])), fig=fig)

# %%
# fig = plt.figure()
x_coef = np.arange(roopcount)
x_dict = np.arange(roopcount)
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
ax = plt.subplot()
# ax.set_ylim(0.001, 3)
# plt.yscale('log')
plt.plot(x_coef, prx_log, color="b", label="X Primal")
plt.plot(x_coef, dux_log, color="g", label="X Dual")
plt.plot(x_dict, prd_log, color="r", label="D Primal")
plt.plot(x_dict, dud_log, color="c", label="D Dual")
plt.title("x" + str(coef_ite) + "d" + str(dict_ite) + "lambd" + str(lamd))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(x_coef, hizero)
ax2.plot(x_coef, goal)
plt.title("x" + str(coef_ite) + "d" + str(dict_ite) +
          "lambd" + str(lamd) + "L0" + str(hizero[ - 1]))
plt.legend()
# %%
plt.plot(x_coef[25:], prx_log[25:], color="b", label="X Primal")
plt.plot(x_dict[25:], prd_log[25:], color="r", label="D Primal")
plt.title("x" + str(coef_ite) + "d" + str(dict_ite) +
          "lambd" + str(lamd) + "roop" + str(roopcount))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
# %%
# np_X = np.array(X).reshape(5, 8, 51, 51).transpose(2, 3, 0, 1)
# plot.imview((x_dictutil.tiledict(np.sum(abs(np_X))), axis=3).squeeze(), cmap=plot.cm.Blues,
#             title='Sparse representation')
# fig.show()
# %%
tesX = []
tesy = []
for i in range(2):
    # x = np.random.normal(0, 1, N * N * M)
    # x = np.zeros(K, N * N * M)
    x = np.zeros(N * N * M)
    tesX.append(x)
for i in range(2):
    x = np.random.normal(0, 1, N * N)
    tesy.append(x)
tesS1 = cv2.resize(np.load("test.npy")[0], (N, N)) / 255
tesS2 = cv2.resize(np.load("test.npy")[1], (N, N)) / 255
tesS = [tesS1.astype(np.float64), tesS2.astype(np.float64)]
sl6, sh6 = util.tikhonov_filter(tesS[0], fltlmbd, npd)
sl7, sh7 = util.tikhonov_filter(tesS[1], fltlmbd, npd)
print(sh1.shape)
S6 = sh6.reshape(N * N)
S7 = sh7.reshape(N * N)
# S6 = tesS[0].reshape(N * N)
# S7 = tesS[1].reshape(N * N)
TESTS = [S6, S7]

# %%
# d_0 = np.random.normal( - 1, 1, (M, d_size, d_size))
X_tes, y_tes, prx_tes, dux_tes, hize_tes = coefficient_learning(
    d, tesX, tesy, TESTS, N * N, M, 1, 0.5, d_size, 220)
# %%
xf_tes = X_to_xf(X_tes, N * N, M)
img_tes = []
img_tes.append(sl6 + reconstruct(xf_tes[0], D, N * N, M))
img_tes.append(sl7 + reconstruct(xf_tes[1], D, N * N, M))
# img_tes.append(reconstruct(xf_tes[0], D, N * N, M))
# img_tes.append(reconstruct(xf_tes[1], D, N * N, M))

# print(sh1)
# print(reconstruct(xf[0], D, N * N, M))

# lambda 0.1 x20 d1 m16 ite50 l0 8941 lambda_tes 0.5 ite _tes 103
#
#   ssim 0.9633069801229223 hizero 6343 psnr 32.48
#   ssim 0.964575618455462 hizero  4748 psnr 30.66
#
# lambda 0.1 x20 d1 m24 ite50 l0 7070 lambda_tes 0.5 ite_tes 140
#
#   ssim 0.9626232868540868 hizero 5954 psnr32.50
#   ssim 0.9657699731773517 hizero 4422 psnr30.91
#
# lambda 0.1 x20 d1 m32 ite50 l0 5470 lambda_tes 0.5 ite_tes 170
#
#   ssim 0.9611599131157266 hizero 5544 psnr32.38
#   ssim 0.9670827170536861 hizero 4252 psnr31.31
#
# lambda 0.02 x1 d1 m16 ite200 l0 4865 lambda_tes 0.5 ite_tes 100
#   ssim 0.9599233062603879 hizero 5449 psnr32.24
#   ssim 0.9653802367166419 hizero 4038 psnr30.84
#
# lambda 0.02 x1 d1 m16 ite50 l0 5709 lambda_tes 0.5 ite_tes 100
#   ssim 0.9589008055344321 hizero 5312 psnr32.05
#   ssim 0.9662954185465689 hizero 4189 psnr30.99
# lambda 0.02 x1 d1 m24 ite50 l0 867 lambda_tes 0.5 ite_tes 135
#    hizero 5578 psnr32.34
#   hizero 4751 psnr31.02
# lambda 0.02 x1 d1 m32 ite50 l0 144 lambda_tes 0.5 ite_tes 165
#    hizero 5936 psnr32.45
#    hizero 5022 psnr30.96

#
# lambda 0.02 x1 d1 m24 ite200 l0 807 lambda_tes 0.5 ite_tes 135
#    hizero 5570 psnr32.26
#    hizero 4742 psnr31.57
# lambda 0.01 x1 d1 m32 ite100 l0 8375 lambda_tes 0.5 ite_tes 170
#    hizero 5449 psnr32.28
#    hizero 4025 psnr30.53
# lambda 0.015 x1 d1 m32 ite50 l0 1500 lambda_tes 0.5 ite_tes 170
#    hizero 5528 psnr32.24
#    hizero 4437 psnr31.09
# lambda 0.01 x1 d1 m32 ite100 l0 6960 lambda_tes 0.5 ite_tes 220
#   ssim 0.9757409055170905 hizero 6368 psnr34.91
#   ssim 0.9776642193089905 hizero 4541 psnr33.21
# 高周波の辞書で元画像のまますると非ゼロ13000 psnr 33
# lambda 0 の辞書では　13035 psnr 33 11323 psnr 31 ite 30
# lambda 0.01 x1 d1 200 m24
# ssim 0.9607905215709278 hizero 5730 psnr 32
# ssim 0.96473573253032 hizero 4384 psnr 30
# random  hizero 6998 psnr 33 hizero 7668 psnr 32
# rho1 lambda 0.5 ite200
# ssim 0.9897960148490316 hizero 9370 psnr 38
# ssim 0.9899393818910095  hizero 7523 psnr 37
# rho1 lambda 0.5 ite 300
# sim 0.9951561146043986 hizero 10961 psnr 41
# ssim 0.9955509738157623 hoizero 9320 psnr 41
# %%
for g in range(len(img_tes)):
    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(tesS[g], title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(img_tes[g], title='Reconstructed[psnr]' +
                str(sm.psnr(tesS[g], img_tes[g])), fig=fig)
    print("ssim", ssim(tesS[g], img_tes[g]))
# %%
print(X_tes[0].shape)
X_tes[0] = X_tes[0].reshape(M, 128, 128)
print(X_tes[0].shape)
X_tes[0] = X_tes[0].transpose(1, 2, 0)
print(X_tes[0].shape)
X_tes[0] = X_tes[0].reshape(128, 128, 1, 1, M)
print(X_tes[0].shape)
# %%
print(X_tes[1].shape)
X_tes[1] = X_tes[1].reshape(M, 128, 128)
print(X_tes[1].shape)
X_tes[1] = X_tes[1].transpose(1, 2, 0)
print(X_tes[1].shape)
X_tes[1] = X_tes[1].reshape(128, 128, 1, 1, M)
print(X_tes[1].shape)
# %%
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(np.sum(abs(X_tes[0]), axis=4).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(np.sum(abs(X_tes[1]), axis=4).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
fig.show()
print("X6 L0:", np.sum(abs(X_tes[0]) != 0))
print("X7 L0:", np.sum(abs(X_tes[1]) != 0))
# %%
x_coef = np.arange(len(prx_tes))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
ax = plt.subplot()
# ax.set_ylim(0.001, 3)
# plt.yscale('log')
plt.plot(x_coef, prx_tes, color="b", label="X Primal")
plt.plot(x_coef, dux_tes, color="g", label="X Dual")
plt.title("x" + str(coef_ite) + "d" + str(dict_ite) + "lambd" + str(lamd))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# ax1.plot(x_coef, hize)
# ax2.plot(x_coef, goal)
plt.title("x" + str(coef_ite) + "d" + str(dict_ite) + "lambd" + str(lamd))
plt.legend()
# %%
d = d.transpose(1, 2, 0)
plot.imview(util.tiledict(d), title='L1 - L1 DR', fgsz=(7, 7))
d = d.transpose(2, 0, 1)
# %%


# %%
import numpy as np
import sporco.prox as pr
import sporco.cnvrep as cnvrep
from sporco import util
from sporco import plot
import matplotlib.pyplot as plt
import sporco.metric as sm
import sporco.linalg as sli
from skimage.metrics import structural_similarity as ssim
import cv2
from typing_extensions import TypedDict


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
        root_N = int(np.sqrt(self.N))
        Xf = self.Xf
        rho = self.rhod
        rho2 = self.rhod * self.rhod
        # self.X = sli.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)
        for _ in range(self.opt["dict_ite"]):
            Raf4E = rho2 * sli.solvemdbi_ism(np.conj(self.Xf), 1/rho, np.fft.rfft(2 * self.E - self.ZE), self.axisK, self.axisM)
            Raf2E = -rho * np.sum(np.conj(Xf) * Raf4E, axis=0, keepdims=True)
            bD = np.fft.rfft(2 * self.D - self.ZD)
            # Raf3 = rho2 * sli.solvemdbi_ism(np.conj(self.Xf), 1/rho, bD, self.axisK, self.axisM)
            # Raf3D = rho * np.sum(np.sum(Xf * Raf3, axis=1, keepdims=True), axis=1, keepdims=True)
            Raf3D = rho2 * sli.solvemdbi_ism(np.conj(self.Xf), 1/rho, np.sum(Xf * bD, axis=1, keepdims=True), self.axisK, self.axisM)
            Raf1D = bD - rho * np.sum(np.conj(Xf) * Raf3D, axis=0)
            self.ZD += np.fft.irfft(Raf1D + Raf2E) - self.D
            self.ZE += np.fft.irfft(Raf3D + Raf4E) - self.E
            self.D = self.pcn(self.ZD)
            self.E = self.ZE - rho * (pr.prox_l1(self.ZE / rho - self.S, 1 / rho) + self.S)
        self.Df = np.fft.rfft(self.D)

        # IXXtf = 1 + rho * rho * np.sum(np.moveaxis(Xf, 1, 0) * Xf.reshape(self.K, self.M, 1, -1), axis=1)
        # for _ in range(self.opt["dict_ite"]):
        #     Raf4E = np.sum(np.fft.rfft(2 * self.E - self.ZE).reshape(1, self.K, -1) / IXXtf, axis=1, keepdims=True)

        #     Raf2E = -rho * np.sum(np.conj(Xf) * Raf4E, axis=0, keepdims=True)
        #     bD = np.fft.rfft(2 * self.D - self.ZD)
        #     Raf3D = rho * np.sum(np.sum(Xf * bD, axis=1, keepdims=True) / IXXtf, axis=1, keepdims=True)
        #     Raf1D = bD - rho * np.sum(np.conj(Xf) * Raf3D, axis=0)
        #     self.ZD += np.fft.irfft(Raf1D + Raf2E) - self.D
        #     self.ZE += np.fft.irfft(Raf3D + Raf4E) - self.E
        #     self.D = self.pcn(self.ZD)
        #     self.E = self.ZE - rho * (pr.prox_l1(self.ZE / rho - self.S, 1 / rho) + self.S)       
        # self.Df = np.fft.rfft(self.D)

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
            self.Y = self.ZY - rho * (pr.prox_l2(self.ZY / rho - self.S, 1 / rho) + self.S)
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
N = int(512 * zoom)
K = 5
# %%
#階層移動
os.chdir("/content/drive/MyDrive/CDL_DR/mydataset/images")
#階層内のPathを全取得
path = glob.glob("*")
training_data = np.empty((1,128,128))
#リサイズ後のサイズ

# IMG_SIZE = 
i = 0
#個別のFilePathに対して処理していきます。
for p in path:
   pathEach = p
  #  print(pathEach) #NumPy配列ndarrayとして読み込まれ、ndarrayを画像として保存
   imageTTTT = cv2.imread(pathEach,cv2.IMREAD_GRAYSCALE)
   if i == 0:
     training_data = imageTTTT.reshape(1, 128, 128)
     i = 1
   else:
     training_data = np.append(training_data, imageTTTT.reshape(1, 128, 128), axis=0)
  #  print(imageTTTT.shape) #画像のりサイズ
  #  img_resize_array = cv2.resize(imageTTTT, (IMG_SIZE, IMG_SIZE))
  #  print(imageTTTT)
  #  print(img_resize_array.dtype)
  #  training_data.append(imageTTTT)
  #  plt.imshow(imageTTTT, cmap="gray")
  #  plt.show()
print(type (training_data))
training_data = np.array(training_data)
print(type(training_data))
print(training_data.shape)
print("=============")
# %%

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

S = ori.reshape(5, N, N)

tesS1 = cv2.resize(np.load("test.npy")[0], (N, N)) / 255
tesS2 = cv2.resize(np.load("test.npy")[1], (N, N)) / 255
tesS = np.asarray([tesS1.astype(np.float64).reshape([-1]), tesS2.astype(np.float64).reshape([-1])]).reshape(2, N, N)
# %%
# 辞書枚数
M = 16
d_size = 12
D0 = np.random.randn(M, d_size, d_size)
# D0 = util.convdicts()['G:12x12x36']
# print(d0.shape)
# D0 = D0.transpose(2, 0, 1)
print(D0.shape)

# %%
lamd = 0.01
rhox = 1
rhod = 1
total_ite = 200
opt = DR_DctLnL1.Options(lmbda=lamd, rhox=rhox, rhod=rhod, total_ite=total_ite, coef_ite=10, dict_ite=10)
# d = DR_DctLnL1(D0, (S - 0.5) * 18, opt)
d = DR_DctLnL1(D0, S, opt)
d.solve()
plot.imview(util.tiledict(d.reconstruct().transpose(1, 2, 0)))
plot.imview(util.tiledict(d.getdict().transpose(1, 2, 0)))
print("L0:", np.sum(d.X != 0, axis=(d.axisM, d.axisN)))
# %%
for i in range(S.shape[0]):
    print("psnr", sm.psnr(ori[i], d.reconstruct()[i]))
# for i in range(S.shape[0]):
#     print("psnr", sm.psnr(sh[i], d.reconstruct()[i]))
# %% TEST_______________
lamd = 0.15
rhox = 1
opt = DR_DctLnL1.Options(lmbda=lamd, rhox=rhox, rhod=rhod, total_ite=total_ite, coef_ite=200, dict_ite=1)
# d = DR_DctLnL1(D0, (S - 0.5) * 18, opt)
tes_se = DR_DctLnL1(d.getdict(), tesS, opt)
tes_se.xstep()
plot.imview(util.tiledict(tes_se.reconstruct().transpose(1, 2, 0)))
plot.imview(util.tiledict(tes_se.getdict().transpose(1, 2, 0)))
print("L0:", np.sum(tes_se.X != 0, axis=(tes_se.axisM, tes_se.axisN)))
for i in range(tesS.shape[0]):
    print("psnr", sm.psnr(tesS[i], tes_se.reconstruct()[i]))
# %%
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

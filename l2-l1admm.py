# %%
from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw   # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.dictlrn import cbpdndl
from sporco import util
from sporco import plot
plot.config_notebook_plotting()
# %%
exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
S1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
S2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
S3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
S4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
S5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
S = np.dstack((S1, S2, S3, S4, S5))
npd = 16
fltlmbd = 5
sl, sh = util.tikhonov_filter(S, fltlmbd, npd)
D0 = np.random.randn(10, 10, 32)
lmbda = 0.1
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns')
d = cbpdndl.ConvBPDNDictLearn(D0, sh, lmbda, opt, dmethod='cns')
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))
D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()
its = d.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.XPrRsdl, its.XDlRsdl, its.DPrRsdl,
          its.DDlRsdl)).T, ptyp='semilogy', xlbl='Iterations',
          ylbl='Residual', lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual'],
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(np.vstack((its.XRho, its.DRho)).T, xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy',
          lgnd=['$\\rho_X$', '$\\rho_D$'], fig=fig)
fig.show()
# %%
plot.imview(util.tiledict(D1),title='L2-L1 ADMM', fgsz=(7, 7))
# %%
# d = D1.transpose(2, 0, 1)
tesX = []
tesy = []
for i in range(2):
    # x = np.random.normal(0, 1, N*N*M)
    # x = np.zeros(K, N*N*M)
    x = np.zeros(N*N*M)
    tesX.append(x)
for i in range(2):
    x = np.random.normal(0, 1, N*N)
    tesy.append(x)
tesS1 = cv2.resize(np.load("test.npy")[0], (N,N))/255
tesS2 = cv2.resize(np.load("test.npy")[1], (N,N))/255
tesS = [tesS1.astype(np.float64), tesS2.astype(np.float64)]
sl6, sh6 = util.tikhonov_filter(tesS[0], fltlmbd, npd)
sl7, sh7 = util.tikhonov_filter(tesS[1], fltlmbd, npd)
print(sh1.shape)
S6 = sh6.reshape(N*N)
S7 = sh7.reshape(N*N)
# S6 = tesS[0].reshape(N*N)
# S7 = tesS[1].reshape(N*N)
TESTS = [S6, S7]

# %%
# d_0 = np.random.normal(-1, 1, (M, d_size, d_size))
from sporco.admm import cbpdn
lmbda = 0.048
opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': 500,
                              'RelStopTol': 5e-3, 'AuxVarObj': False})
b0 = cbpdn.ConvBPDN(D1, sh6, lmbda, opt, dimK=0)
X6 = b0.solve()
b1 = cbpdn.ConvBPDN(D1, sh7, lmbda, opt, dimK=0)
X7 = b1.solve()
shr6 = b0.reconstruct().squeeze()
imgr6 = sl6 + shr6
print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(tesS[0], imgr6))
shr7 = b1.reconstruct().squeeze()
imgr7 = sl7 + shr7
print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(tesS[1], imgr7))
# %%
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(np.sum(abs(X6), axis=b0.cri.axisM).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(np.sum(abs(X7), axis=b1.cri.axisM).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
fig.show()
print("X6 L0:",np.sum(abs(X6)!=0))
print("X7 L0:",np.sum(X7!=0))
print(X6.shape)
print(type(X6))
# %%
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(tesS[0], title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(imgr6, title='Reconstructed'+str(sm.psnr(tesS[0], imgr6)), fig=fig)
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(tesS[1], title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(imgr7, title='Reconstructed'+str(sm.psnr(tesS[1], imgr7)), fig=fig)
fig.show()

# %%
d1 = D1.transpose(2,0,1)
print(D1.shape)
print(d1.shape)
# %%
X_tes, y_tes, prx_tes, dux_tes, hize_tes = coefficient_learning(d1, tesX, tesy, TESTS, N*N, M, 1, 0.5, d_size, 200)
xf_tes = X_to_xf(X_tes, N*N, M)
DD1 = D_to_d(d1, N*N, d_size)
img_tes = []
img_tes.append(sl6+reconstruct(xf_tes[0], DD1, N*N, M))
img_tes.append(sl7+reconstruct(xf_tes[1], DD1, N*N, M))
for g in range(len(img_tes)):
    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(tesS[g], title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(img_tes[g], title='Reconstructed[psnr]'+str(sm.psnr(tesS[g], img_tes[g])), fig=fig)
    print("ssim", ssim(tesS[g], img_tes[g]))

# %%
print(X_tes[0].shape)
X_tes[0] = X_tes[0].reshape(32, 128, 128)
print(X_tes[0].shape)
X_tes[0] = X_tes[0].transpose(1, 2, 0)
print(X_tes[0].shape)
X_tes[0] = X_tes[0].reshape(128, 128, 1, 1, 32)
print(X_tes[0].shape)
# %%
print(X_tes[1].shape)
X_tes[1] = X_tes[1].reshape(32, 128, 128)
print(X_tes[1].shape)
X_tes[1] = X_tes[1].transpose(1, 2, 0)
print(X_tes[1].shape)
X_tes[1] = X_tes[1].reshape(128, 128, 1, 1, 32)
print(X_tes[1].shape)
# %%
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(np.sum(abs(X_tes[0]), axis=b0.cri.axisM).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(np.sum(abs(X_tes[1]), axis=b1.cri.axisM).squeeze(), cmap=plot.cm.Blues,
            title='Sparse representation', fig=fig)
fig.show()
print("X6 L0:",np.sum(abs(X_tes[0])!=0))
print("X7 L0:",np.sum(abs(X_tes[1])!=0))
print(X6.shape)
print(type(X6))
# %%
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(tesS[0], title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(imgr6, title='Reconstructed'+str(sm.psnr(tesS[0], imgr6)), fig=fig)
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(tesS[1], title='Original', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(imgr7, title='Reconstructed'+str(sm.psnr(tesS[1], imgr7)), fig=fig)
fig.show()

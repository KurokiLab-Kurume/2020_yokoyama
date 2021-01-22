# %%
import numpy as np
import sporco.prox as pr
import sporco.cnvrep as cnvrep
from sporco import util
from sporco import plot
import matplotlib.pyplot as plt
import sporco.metric as sm
import os
import cv2
import glob
from skimage.metrics import structural_similarity as ssim
# %%

def zero_pad(D, N, d_size):
    N = int(np.sqrt(N))
    return np.pad(D, ((0, N-d_size), (0, N-d_size)), 'constant')


def D_to_df(D, N, d_size):
    flat = np.zeros(N*D.shape[0])
    for i in range(D.shape[0]):
        a = zero_pad(D[i], N, d_size)
        flat[i*N:(i+1)*N] = np.ravel(a)
    # emptyの方がメモリ抑えられるかも
    df = np.zeros(N*D.shape[0], dtype=np.complex)
    for i in range(D.shape[0]):
        df[i*N:(i+1)*N] = np.fft.fft(flat[i*N:(i+1)*N])
    return df


# def D_to_df_kai(D, N, M, d_size):
#     dd = d_size*d_size
#     d = np.zeros(N*M)
#     df = np.zeros(N*M, dtype=np.complex)
#     for i in range(M):
#         flat = np.ravel(D[i])
#         d[i*N:(i+1)*N] = np.pad(flat, (0, N-dd))
#         df[i*N:(i+1)*N] = np.fft.fft(d[i*N:(i+1)*N])
#     return d, df


def D_to_d(D, N, d_size):
    flat = np.zeros(N*D.shape[0])
    for i in range(D.shape[0]):
        a = zero_pad(D[i], N, d_size)
        flat[i*N:(i+1)*N] = np.ravel(a)
    return flat


def make_IDDt(df, N, M, alpha):
    DDt = np.zeros(N, dtype=np.complex)
    for i in range(M):
        DDt = df[i*N:(i+1)*N]*df[i*N:(i+1)*N].conjugate()+DDt
    return 1 + alpha*alpha*DDt


# def make_IXXt(XF, N, M, alpha):
#     XXt = np.zeros(N, dtype=np.complex)
#     xxt = np.zeros(N, dtype=np.complex)
#     for xf in XF:
#         for i in range(M):
#             xxt = xf[i*N:(i+1)*N]*xf[i*N:(i+1)*N].conjugate()+xxt
#         XXt = xxt + XXt
#     return 1 + alpha*alpha*XXt


def make_migi(IDDt, df, byf, N, M, alpha):
    Dtb = np.zeros(N*M, dtype=np.complex)
    migi_NM = np.zeros(N*M)
    migi_N = np.fft.ifft(byf/IDDt)
    migi_N = migi_N.astype(np.float64)
    for i in range(M):
        Dtb[i*N:(i+1)*N] = df[i*N:(i+1)*N].conjugate()/IDDt*byf*alpha*-1
    for i in range(M):
        migi_NM[i*N:(i+1)*N] = np.fft.ifft(Dtb[i*N:(i+1)*N])
    migi_NM = migi_NM.astype(np.float64)
    return np.concatenate([migi_NM, migi_N], 0)


def make_hidari(IDDt, df, bxf, N, M, alpha):
    DtDb = np.zeros(N*M, dtype=np.complex)
    Db = np.zeros(N, dtype=np.complex)
    for i in range(M):
        Db = df[i*N:(i+1)*N]*bxf[i*N:(i+1)*N] + Db
    hidari_N = alpha*Db/IDDt
    for i in range(M):
        DtDb[i*N:(i+1)*N] = df[i*N:(i+1)*N].conjugate()*hidari_N
    DtDb = bxf - (alpha)*DtDb
    hidari_N = np.fft.ifft(hidari_N)
    hidari_NM = np.zeros(N*M)
    for i in range(M):
        hidari_NM[i*N:(i+1)*N] = np.fft.ifft(DtDb[i*N:(i+1)*N])
    hidari_N = hidari_N.astype(np.float64)
    hidari_NM = hidari_NM.astype(np.float64)
    return np.concatenate([hidari_NM, hidari_N], 0)


# def resolvent_b(df, xy, s, rho, lamd, N, M):
#     x = pr.prox_l1(xy[:N*M], lamd/rho)
#     y = xy[N*M:] - (pr.prox_l1(xy[N*M:] - s, 1/rho) + s)
#     return np.concatenate([x, y], 0)


def make_y(df, x, N, M):
    # yはh(Dx)=l1(Dx-s)よりy = Dx
    # print("##### in make_xy #####")
    # print("x.shape :", x.shape)
    xf = np.zeros(x.shape, dtype=np.complex)
    for i in range(M):
        xf[i*N:(i+1)*N] = np.fft.fft(x[i*N:(i+1)*N])
    yf = np.zeros(N, dtype=np.complex)
    for i in range(M):
        yf = df[i*N:(i+1)*N]*xf[i*N:(i+1)*N] + yf
    y = np.fft.ifft(yf)
    y = y.astype(np.float64)
    return y


def D_to_dd_conj(D, dy, xf, N, d_size):
    flat = np.zeros(N*D.shape[0])
    for i in range(D.shape[0]):
        a = zero_pad(D[i], N, d_size)
        flat[i*N:(i+1)*N] = np.ravel(a)
    # emptyの方がメモリ抑えられるかも
    df = np.zeros(N*N*D.shape[0], dtype=np.complex)
    for i in range(D.shape[0]):
        df[i*N:(i+1)*N] = np.fft.fft(flat[i*N:(i+1)*N])
    # yf = np.zeros(N, dtype=np.complex)
    # for i in range(D.shape[0]):
    #     yf = df[i*N:(i+1)*N]*xf[i*N:(i+1)*N] + yf
    # y = np.fft.ifft(yf)*np.sqrt(N)
    # y = y.astype(np.float64)
    return np.concatenate([flat, dy], 0)


def coefficient_learning(D, x, y, s, N, M, rho, lamd, d_size, ite=200):
    df = D_to_df(D, N, d_size)
    reX = []
    s_count = 0
    xlog = []
    ylog = []
    zero = []
    for j in range(len(x)):
        s_count = s_count+1
        # y = make_y(df, x[j], N, M)
        # y = np.random.normal(0, 1, N*N)
        count = 0
        while True:
            pr_x = pr.prox_l1(x[j], lamd/rho)
            pr_y = y[j] - (pr.prox_l1(y[j] - s[j], 1/rho) + s[j])
            bx = 2 * pr_x - x[j]
            by = 2 * pr_y - y[j]
            bxf = np.zeros(bx.shape, dtype=np.complex)
            for i in range(M):
                bxf[i*N:(i+1)*N] = np.fft.fft(bx[i*N:(i+1)*N])
            byf = np.fft.fft(by)
            IDDt = make_IDDt(df, N, M, rho)
            hidari = make_hidari(IDDt, df, bxf, N, M, rho)
            migi = make_migi(IDDt, df, byf, N, M, rho)
            xy = np.concatenate([x[j], y[j]], 0)
            pr_xy = np.concatenate([pr_x, pr_y], 0)
            xy = xy + migi + hidari - pr_xy
            # rsdl_n = max(np.linalg.norm(migi[:N*M]+hidari[:N*M]), np.linalg.norm(pr_xy[:N*M]))
            # rsdl_ny = max(np.linalg.norm(migi[N*M:]+hidari[N*M:]), np.linalg.norm(pr_xy[N*M:]))
            # move = np.linalg.norm(migi[:N*M]+hidari[:N*M]-pr_xy[:N*M])/rsdl_n
            x[j] = xy[:N*M]
            y[j] = xy[N*M:]
            move = pr.prox_l1(x[j], lamd/rho) - pr_x
            xlog.append(np.linalg.norm(move))
            # move = np.linalg.norm(migi[N*M:]+hidari[N*M:]-pr_xy[N*M:])/rsdl_ny
            move = y[j] - (pr.prox_l1(y[j] - s[j], 1/rho) + s[j]) - pr_y
            ylog.append(np.linalg.norm(move))
            h = pr.prox_l1(x[j], lamd/rho)
            hizero = np.linalg.norm(h.astype(np.float64), ord=0)
            # a, hizero = check(df, xy, s, N, M, lamd)
            count = count + 1
            print("count :", count, "move["+str(s_count)+"]", np.linalg.norm(move), "hizero", hizero)
            if count >= ite:
                break
        zero.append(hizero)
        # print("final gosa", gosa)
        print("final coefcount :", count)
        reX.append(h)
    return reX, y, xlog[:ite], ylog[:ite], sum(zero)/len(zero)


def X_to_xf(X, N, M):
    reXF = []
    for j in range(len(X)):
        xf = np.zeros(N*M, dtype=np.complex)
        for i in range(M):
            xf[i*N:(i+1)*N] = np.fft.fft(X[j][i*N:(i+1)*N])
        reXF.append(xf)
    return reXF


def resolvent_d(d, y, s, rho, N, M, d_size):
    rootN = int(np.sqrt(N))
    d = d.reshape(M, rootN, rootN)
    d = d.transpose(1, 2, 0)
    d_Pcn = cnvrep.Pcn(d.reshape(rootN, rootN, 1, 1, M), (d_size, d_size, M), Nv=(rootN, rootN)).squeeze()
    d_Pcn = d_Pcn.transpose(2, 0, 1)
    d = d_Pcn.reshape(N*M)
    y = y - (pr.prox_l1(y - s, 1/rho) + s)
    return np.concatenate([d, y], 0)


def pr_d(d, N, M, d_size):
    rootN = int(np.sqrt(N))
    d = d.reshape(M, rootN, rootN)
    d = d.transpose(1, 2, 0)
    d_Pcn = cnvrep.Pcn(d.reshape(rootN, rootN, 1, 1, M), (d_size, d_size, M), Nv=(rootN, rootN)).squeeze()
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
            re_d = resolvent_d(dd_conj[j][:N*M], dd_conj[j][N*M:], s[j], rho, N, M, d_size)
            b = 2 * re_d - dd_conj[j]
            bf = np.zeros(b.shape, dtype=np.complex)
            for k in range(M+1):
                bf[k*N:(k+1)*N] = np.fft.fft(b[k*N:(k+1)*N])
            IXXt = make_IDDt(xf[j], N, M, rho)
            hidari = make_hidari(IXXt, xf[j], bf[:N*M], N, M, rho)
            migi = make_migi(IXXt, xf[j], bf[N*M:], N, M, rho)
            dd_conj[j] = dd_conj[j] + migi + hidari - re_d
            # rsdl_n = max(np.linalg.norm(migi[:N*M]+hidari[:N*M]), np.linalg.norm(re_d[:N*M]))
            # rsdl_nd = max(np.linalg.norm(migi[N*M:]+hidari[N*M:]), np.linalg.norm(re_d[N*M:]))
            # pmove = np.linalg.norm(migi[:N*M]+hidari[:N*M]-re_d[:N*M])/rsdl_n + pmove
            # dmove = np.linalg.norm(migi[N*M:]+hidari[N*M:]-re_d[N*M:])/rsdl_nd + dmove 
            # move = np.linalg.norm(migi+hidari-re_d)/rsdl_n + move
            # print("move["+str(j+1)+"] : ", np.linalg.norm(migi+hidari-re_d)/rsdl_n)
            pmove = np.linalg.norm(dd_conj[j][:N*M]-re_d[:N*M]) + pmove
            dmove = np.linalg.norm(dd_conj[j][N*M:]-re_d[N*M:]) + dmove
        prlog.append(pmove/len(dd_conj))
        dulog.append(dmove/len(dd_conj))
        d_mean = sum(dd_conj)/len(dd_conj)
        # d = pr_d(d_mean[:N*M], N, M, d_size)
        for kk in range(len(dd_conj)):
            dd_conj[kk][:N*M] = d_mean[:N*M]
        # gosa, hizero = check(xf, dd_conj, s, N, M, lamd)
        count = count + 1
        print("count :", count)
        if count >= ite:
            break
    print("final dic count :", count)
    d = pr_d(d_mean[:N*M], N, M, d_size)
    return d, d_mean[N*M:], prlog, dulog


def reconstruct(xf, d, N, M):
    df = np.zeros(N*M, dtype=np.complex)
    sf = np.zeros(N, dtype=np.complex)
    s = np.zeros(N*M)
    for i in range(M):
        df[i*N:(i+1)*N] = np.fft.fft(d[i*N:(i+1)*N])
    for i in range(M):
        sf = xf[i*N:(i+1)*N] * df[i*N:(i+1)*N] + sf
    s = np.fft.ifft(sf)
    s = s.astype(np.float64)
    sqrtN = int(np.sqrt(N))
    return s.reshape(sqrtN, sqrtN)


def dx_s(D, xf, s, N, M, K, d_size):
    samu = 0.
    df = D_to_df(D, N, d_size)
    for k in range(K):
        dxf = np.zeros(N, dtype=np.complex)
        for i in range(M):
            dxf = df[i*N:(i+1)*N]*xf[k][i*N:(i+1)*N] + dxf
        dx = np.fft.ifft(dxf)
        dx = dx.astype(np.float64)
        dx_s = dx - s[k]
        samu = np.linalg.norm(dx_s, ord=1) + samu
    return samu


def l1x(X, N, M):
    samu = 0
    for i in range(len(X)):
        for j in range(M):
            samu = np.linalg.norm(X[i][j*N:(j+1)*N], ord=1) + samu
    return samu
        

# %%
# #階層移動
# os.chdir("/Users/yokoyama/Desktop/2020_yokoyama/images")
# #階層内のPathを全取得
# path = glob.glob("*")
# training_data = np.empty((1,128,128))
# #リサイズ後のサイズ

# IMG_SIZE = 128
# i = 0
# #個別のFilePathに対して処理していきます。
# for p in path:
#    pathEach = p
#   #  print(pathEach) #NumPy配列ndarrayとして読み込まれ、ndarrayを画像として保存
#    imageTTTT = cv2.imread(pathEach, cv2.IMREAD_GRAYSCALE)
#    if i == 0:
#      training_data = cv2.resize(imageTTTT, (IMG_SIZE, IMG_SIZE))
#      i = 1
#    else:
#      training_data = np.append(training_data, cv2.resize(imageTTTT, (IMG_SIZE, IMG_SIZE)), axis=0)
# fig = plot.figure(figsize=(14, 7))
# plot.imview(training_data[0], title='Original', fig=fig)


# %%


exim = util.ExampleImages(scaled=True, zoom=0.25, gray=True)
s1 = exim.image('barbara.png', idxexp=np.s_[10:522, 100:612])
s2 = exim.image('kodim23.png', idxexp=np.s_[:, 60:572])
s3 = exim.image('monarch.png', idxexp=np.s_[:, 160:672])
s4 = exim.image('sail.png', idxexp=np.s_[:, 210:722])
s5 = exim.image('tulips.png', idxexp=np.s_[:, 30:542])
ori = [s1, s2, s3, s4, s5]
# ori = [s1]
npd = 16
fltlmbd = 10
sl1, sh1 = util.tikhonov_filter(s1, fltlmbd, npd)
sl2, sh2 = util.tikhonov_filter(s2, fltlmbd, npd)
sl3, sh3 = util.tikhonov_filter(s3, fltlmbd, npd)
sl4, sh4 = util.tikhonov_filter(s4, fltlmbd, npd)
sl5, sh5 = util.tikhonov_filter(s5, fltlmbd, npd)
print(s1.shape)
N = sh1.shape[0]
S1 = sh1.reshape(N*N)
S2 = sh2.reshape(N*N)
S3 = sh3.reshape(N*N)
S4 = sh4.reshape(N*N)
S5 = sh5.reshape(N*N)
S = [S1, S2, S3, S4, S5]
# S = [S1]
print(S1.shape)
# 辞書枚数
M = 20
d_size = 10
# proxのstep
# D = util.convdicts()['G:12x12x36']
# D = D.transpose(2, 0, 1)
K = len(S)
d = np.random.normal(0, 1, (M, d_size, d_size))
X = []
y = []
roopcount = 0
for i in range(K):
    # x = np.random.normal(0, 1, N*N*M)
    # x = np.zeros(K, N*N*M)
    x = np.zeros(N*N*M)
    X.append(x)
for i in range(K):
    x = np.random.normal(0, 1, N*N)
    y.append(x)
dcon = np.random.normal(0, 1, N*N)
prx_log = []
dux_log = []
prd_log = []
dud_log = []
hizero = []
goal = []
# %%
rhox = 1
rhod = 1
lamd = 0.1
coef_ite = 40
dict_ite = 1
total_ite = 50
# %%
for i in range(total_ite):
    roopcount = roopcount + 1
    X, y, prx, dux, hize = coefficient_learning(d, X, y, S, N*N, M, rhox, lamd, d_size, coef_ite)
    prx_log.append(prx[coef_ite-1])
    dux_log.append(dux[coef_ite-1])
    hizero.append(hize)
    xf = X_to_xf(X, N*N, M)
    d, dcon, prd, dud = dictinary_learning(d, dcon, xf, S, N*N, M, rhod, d_size, dict_ite)
    prd_log.append(prd[0])
    dud_log.append(dud[0])
    d = pr_d(d, N*N, M, d_size)
    # X = X.transpose(1, 2, 0)
    d = d[:, :d_size, :d_size]
    if i == total_ite-1:
        d = d.transpose(1, 2, 0)
        plot.imview(util.tiledict(d), fgsz=(7, 7))
        d = d.transpose(2, 0, 1)
    mindx_s = dx_s(d, xf, S, N*N, M, K, d_size) + lamd*l1x(X, N*N, M)
    goal.append(mindx_s)
    D = D_to_d(d, N*N, d_size)
    image = reconstruct(xf[0], D, N*N, M)
    print("iterate number: ", roopcount)
    imgr = sl1 + image
    print("Reconstruction PSNR: %.2fdB\n" % sm.psnr(ori[0], imgr))
    # fig = plot.figure(figsize=(14, 7))
    # plot.subplot(1, 2, 1)
    # plot.imview(ori[0], title='Original', fig=fig)
    # plot.subplot(1, 2, 2)
    # plot.imview(image/N*10 + sl1, title='Reconstructed['+str(i+1)+"]"+str(sm.psnr(ori[0], imgr)), fig=fig)
    # for j in range(len(xf)):
    #     image = reconstruct(xf[j], D, N*N, M)
    #     fig = plot.figure(figsize=(14, 7))
    #     plot.subplot(1, 2, 1)
    #     plot.imview(ori[j], title='Original', fig=fig)
    #     plot.subplot(1, 2, 2)
    #     plot.imview(image, title='Reconstructed['+str(i+1)+"]", fig=fig)
# %%
print(sh1[0][0]/image[0][0])
print(sh1)
print(image)
print(np.array(X).shape)
# %%
imgr = []
imgr.append(sl1 + reconstruct(xf[0], D, N*N, M))
imgr.append(sl2 + reconstruct(xf[1], D, N*N, M))
imgr.append(sl3 + reconstruct(xf[2], D, N*N, M))
imgr.append(sl4 + reconstruct(xf[3], D, N*N, M))
imgr.append(sl5 + reconstruct(xf[4], D, N*N, M))
# imgr.append( reconstruct(xf[0], D, N*N, M))
# imgr.append( reconstruct(xf[1], D, N*N, M))
# imgr.append(reconstruct(xf[2], D, N*N, M))
# imgr.append( reconstruct(xf[3], D, N*N, M))
# imgr.append( reconstruct(xf[4], D, N*N, M))
for g in range (K):
    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(ori[g], title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(imgr[g], title='Reconstructed['+str(i+1)+"]"+str(sm.psnr(ori[g], imgr[g])), fig=fig)
    # print(measure.compare_ssim(imgr[g], ori[g]))
# fig = plot.figure(figsize=(14, 7))
# plot.subplot(1, 2, 1)
# plot.imview(sh1, title='Original', fig=fig)
# plot.subplot(1, 2, 2)
# plot.imview(imgr[0], title='Reconstructed['+str(i+1)+"]"+str(sm.psnr(sh1, imgr[0])), fig=fig)

# %%
# fig = plt.figure()
x_coef = np.arange(roopcount)
x_dict = np.arange(roopcount)
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
ax=plt.subplot()
# ax.set_ylim(0.001, 3)
# plt.yscale('log')
plt.plot(x_coef, prx_log, color = "b", label = "X Primal")
plt.plot(x_coef, dux_log, color = "g", label = "X Dual")
plt.plot(x_dict, prd_log, color = "r", label = "D Primal")
plt.plot(x_dict, dud_log, color = "c", label = "D Dual")
plt.title("x"+str(coef_ite)+"d"+str(dict_ite)+"lambd"+str(lamd))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(x_coef, hizero)
ax2.plot(x_coef, goal)
plt.title("x"+str(coef_ite)+"d"+str(dict_ite)+"lambd"+str(lamd))
plt.legend()
# %%
plt.plot(x_coef[25:], prx_log[25:], color = "b", label = "X Primal")
plt.plot(x_dict[25:], prd_log[25:], color = "r", label = "D Primal")
plt.title("x"+str(coef_ite)+"d"+str(dict_ite)+"lambd"+str(lamd)+"roop"+str(roopcount))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
# %%
np_X = np.array(X).reshape(5, 8, 51, 51).transpose(2, 3, 0, 1)
plot.imview((x_dictutil.tiledict(np.sum(abs(np_X))), axis=3).squeeze(),cmap=plot.cm.Blues,
            title='Sparse representation')
fig.show()
# %%
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
TESTS = [S6, S7]

# %%
# d_0 = np.random.normal(-1, 1, (M, d_size, d_size))
X_tes, y_tes, prx_tes, dux_tes, hize_tes = coefficient_learning(d, tesX, tesy, TESTS, N*N, M, 1, 0.5, d_size, 100)
# %%
xf_tes = X_to_xf(X_tes, N*N, M)
img_tes = []
img_tes.append(sl6+reconstruct(xf_tes[0], D, N*N, M))
img_tes.append(sl7+reconstruct(xf_tes[1], D, N*N, M))
# %%
print(sh1)
print(reconstruct(xf[0], D, N*N, M))
# %%
for g in range(len(img_tes)):
    fig = plot.figure(figsize=(14, 7))
    plot.subplot(1, 2, 1)
    plot.imview(tesS[g], title='Original', fig=fig)
    plot.subplot(1, 2, 2)
    plot.imview(img_tes[g], title='Reconstructed['+str(i+1)+"]"+str(sm.psnr(tesS[g], img_tes[g])), fig=fig)
# %%
x_coef = np.arange(200)
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
ax=plt.subplot()
# ax.set_ylim(0.001, 3)
# plt.yscale('log')
plt.plot(x_coef, prx, color = "b", label = "X Primal")
plt.plot(x_coef, dux, color = "g", label = "X Dual")
plt.title("x"+str(coef_ite)+"d"+str(dict_ite)+"lambd"+str(lamd))
plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.legend()
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# ax1.plot(x_coef, hize)
# ax2.plot(x_coef, goal)
plt.title("x"+str(coef_ite)+"d"+str(dict_ite)+"lambd"+str(lamd))
plt.legend()
# %%
d = d.transpose(1, 2, 0)
plot.imview(util.tiledict(d), fgsz=(7, 7))
d = d.transpose(2, 0, 1)
# %%

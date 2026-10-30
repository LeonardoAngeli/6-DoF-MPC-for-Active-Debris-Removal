import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

rng = np.random.default_rng(7)
W, H = 16, 7
fig = plt.figure(figsize=(W, H), dpi=150)

# ---- Background: space gradient + Earth limb + stars (2D layer) ----
bg = fig.add_axes([0, 0, 1, 1])
bg.set_axis_off()
ny, nx = 700, 1600
yy, xx = np.mgrid[0:ny, 0:nx]
img = np.zeros((ny, nx, 3))
img[..., 0] = 0.01; img[..., 1] = 0.015; img[..., 2] = 0.04
# Earth: large circle centred below the frame
cx, cy, R = nx * 0.40, -ny * 2.45, ny * 2.75
d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
earth = d < R
shade = np.clip((R - d) / (0.35 * ny), 0, 1)
img[earth, 0] = 0.02 + 0.10 * shade[earth]
img[earth, 1] = 0.06 + 0.22 * shade[earth]
img[earth, 2] = 0.16 + 0.40 * shade[earth]
# Atmosphere glow
glow = np.exp(-((d - R) / 10.0)**2) * (d > R - 30)
img[..., 0] += 0.25 * glow; img[..., 1] += 0.55 * glow; img[..., 2] += 1.0 * glow
img = np.clip(img, 0, 1)
bg.imshow(img, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
# Stars above the limb
sx, sy = rng.uniform(0, 1, 900), rng.uniform(0, 1, 900)
above = np.sqrt((sx * nx - cx)**2 + (sy * ny - cy)**2) > R + 15
bg.scatter(sx[above], sy[above], s=rng.uniform(0.1, 2.5, above.sum()),
           c="white", alpha=rng.uniform(0.3, 1, above.sum()), linewidths=0)
bg.set_xlim(0, 1); bg.set_ylim(0, 1)

# ---- 3D scene (LVLH, metres): X = along-track y, Y = cross-track z, Z = radial x ----
ax = fig.add_axes([0.22, -0.25, 0.85, 1.45], projection="3d")
ax.set_facecolor((0, 0, 0, 0))
ax.set_axis_off()

def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)

def shaded(ax, X, Y, Z, base, normals):
    inten = np.clip(np.einsum("ijk,k->ij", normals, light), 0, 1) * 0.75 + 0.25
    cols = np.empty(X.shape + (4,))
    for k in range(3):
        cols[..., k] = np.clip(base[k] * inten, 0, 1)
    cols[..., 3] = 1.0
    ax.plot_surface(X, Y, Z, facecolors=cols[:-1, :-1] if False else cols, linewidth=0,
                    antialiased=True, shade=False, rstride=1, cstride=1)

# SL-16 upper stage: cylinder (L = 11 m, D = 3.9 m) + nozzle, tilted (tumbling)
Rt = rot(0.5, 0.35, 0.9)
th = np.linspace(0, 2 * np.pi, 60)
L, r = 11.0, 1.95
zz = np.linspace(-L / 2, L / 2, 25)
TH, ZZ = np.meshgrid(th, zz)
P = np.stack([r * np.cos(TH), r * np.sin(TH), ZZ], -1)
N = np.stack([np.cos(TH), np.sin(TH), 0 * TH], -1)
P, N = P @ Rt.T, N @ Rt.T
shaded(ax, P[..., 0], P[..., 1], P[..., 2], (0.80, 0.80, 0.78), N)
# Interstage band (dark)
zb = np.linspace(L / 2 - 1.2, L / 2, 6)
THb, ZB = np.meshgrid(th, zb)
Pb = np.stack([1.0 * r * np.cos(THb) * 1.01, 1.01 * r * np.sin(THb), ZB], -1) @ Rt.T
Nb = np.stack([np.cos(THb), np.sin(THb), 0 * THb], -1) @ Rt.T
shaded(ax, Pb[..., 0], Pb[..., 1], Pb[..., 2], (0.35, 0.40, 0.35), Nb)
# Nozzle (cone)
zn = np.linspace(0, 2.6, 12)
THn, ZN = np.meshgrid(th, zn)
rn = 0.4 + 0.45 * ZN / 2.6
Pn = np.stack([rn * np.cos(THn), rn * np.sin(THn), -L / 2 - ZN], -1) @ Rt.T
Nn = np.stack([np.cos(THn), np.sin(THn), 0.3 + 0 * THn], -1) @ Rt.T
shaded(ax, Pn[..., 0], Pn[..., 1], Pn[..., 2], (0.30, 0.28, 0.27), Nn)

# Keep-out sphere (wireframe)
u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
Rko = 20.0
ax.plot_wireframe(Rko * np.cos(u) * np.sin(v), Rko * np.sin(u) * np.sin(v), Rko * np.cos(v),
                  color=(1.0, 0.35, 0.3), alpha=0.18, linewidth=0.5)

# Approach trajectories: co-elliptic arrival, bending onto V-bar at y = -30 m
def approach(dh, z0, y0):
    s = np.linspace(0, 1, 300)
    yv = y0 + (-30 - y0) * (1 - (1 - s)**2.2)
    xr = -dh * (1 - s)**1.6 + 6 * np.sin(np.pi * s) * (1 - s)
    zc = z0 * (1 - s)**2
    return yv, zc, xr

for k in range(14):
    yv, zc, xr = approach(rng.uniform(20, 40), rng.normal(0, 8), rng.uniform(-110, -80))
    ax.plot(yv, zc, xr, color=(0.3, 0.85, 1.0), alpha=0.35, linewidth=0.9)
yv, zc, xr = approach(40, 8, -100)
ax.plot(yv, zc, xr, color=(0.45, 0.95, 1.0), linewidth=2.2)

# Chaser at the hold point (bus + solar panels), with a short thruster plume
cy_, cz_, cx_ = -30.0, 0.0, 0.0
b = 1.0
for (dx, dy, dz) in [(0, 0, 0)]:
    corners = np.array([[i, j, k] for i in (-b, b) for j in (-b, b) for k in (-b, b)], float)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def box(c, sx, sy, sz, color):
    x0, y0, z0 = c
    V = np.array([[x0 + i * sx, y0 + j * sy, z0 + k * sz] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    F = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([V[f] for f in F], facecolor=color, edgecolor=(0.9, 0.8, 0.4, 0.6), linewidth=0.3))
box((cy_, cz_, cx_), 1.0, 1.0, 1.0, (0.85, 0.65, 0.20))            # gold MLI bus
box((cy_, cz_ + 3.6, cx_), 0.08, 2.5, 0.9, (0.10, 0.18, 0.45))     # solar panel
box((cy_, cz_ - 3.6, cx_), 0.08, 2.5, 0.9, (0.10, 0.18, 0.45))     # solar panel
pl = np.linspace(0, 1, 20)
ax.plot(cy_ - 1.0 - 4 * pl, 0 * pl, 0 * pl, color=(0.7, 0.9, 1.0), alpha=0.5, linewidth=3)

# V-bar guide
ax.plot([-115, 25], [0, 0], [0, 0], color="white", alpha=0.12, linewidth=0.8, linestyle="--")

ax.set_xlim(-95, 25); ax.set_ylim(-35, 35); ax.set_zlim(-40, 25)
ax.set_box_aspect((120, 70, 65))
ax.view_init(elev=18, azim=-62)

# ---- Title ----
fig.text(0.04, 0.86, "MPC for Active Debris Removal", color="white", fontsize=30,
         fontweight="bold", family="DejaVu Sans")
fig.text(0.04, 0.79, "Autonomous rendezvous with an uncooperative SL-16 upper stage",
         color=(0.75, 0.85, 1.0), fontsize=15, family="DejaVu Sans")

fig.savefig("cover.png", dpi=150, facecolor=(0.01, 0.015, 0.04))
print("saved")

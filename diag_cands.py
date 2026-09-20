import sys
import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, "/workspace")
import segment_board as sb
from colorize_pointcloud import load_camera_info, load_extrinsics, rotation_z

K, dist = load_camera_info("/workspace/config/camera_info.yaml")
R, t = load_extrinsics("/workspace/config/camera_extrinsics.yaml")
Rf = R @ rotation_z(90.0)
rvec = cv2.Rodrigues(Rf)[0]


def window_quad(pts, L=0.84, W=0.60):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = np.cross(vt[0], vt[1])
    n /= np.linalg.norm(n)
    e1, e2 = vt[0], vt[1]
    u, v = (pts - c) @ e1, (pts - c) @ e2
    best = (-1, 0, 0, 0)
    for a in np.arange(-90, 90.1, 3.0):
        r = np.radians(a)
        ca, sa = np.cos(r), np.sin(r)
        uu = u * ca + v * sa
        vv = -u * sa + v * ca
        ulo, vlo = uu.min() - 0.03, vv.min() - 0.03
        nu = int((uu.max() - ulo) / 0.01) + 3
        nv = int((vv.max() - vlo) / 0.01) + 3
        G = np.zeros((nv, nu))
        iu = np.clip(((uu - ulo) / 0.01).astype(int), 0, nu - 1)
        iv = np.clip(((vv - vlo) / 0.01).astype(int), 0, nv - 1)
        np.add.at(G, (iv, iu), 1.0)
        I = np.pad(G.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        kw, kh = 84, 60
        for rr in range(0, nv - kh):
            row = I[rr + kh, kw:] - I[rr, kw:] - I[rr + kh, :-kw] + I[rr, :-kw]
            j = int(np.argmax(row))
            if row[j] > best[0]:
                best = (float(row[j]), ulo + (j + kw / 2) * 0.01, vlo + (rr + kh / 2) * 0.01, a)
    cnt, wc, hc, ang = best
    th = np.radians(ang)
    ca, sa = np.cos(th), np.sin(th)
    corners = []
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        du = su * L / 2
        dv = sv * W / 2
        uu = du * ca - dv * sa
        vv = du * sa + dv * ca
        corners.append(c + (wc + uu) * e1 + (hc + vv) * e2)
    return np.array(corners), cnt / len(pts)


d = sys.argv[1]
path = f"/workspace/turntable_output/20260918_{d}"
with open(path + "/merged.ply") as f:
    for i, l in enumerate(f, 1):
        if l.strip() == "end_header":
            break
P = np.loadtxt(path + "/merged.ply", skiprows=i)[:, :3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P)
ds = np.asarray(pcd.voxel_down_sample(0.01).points)
cands = [(n / np.linalg.norm(n), dd, idx) for n, dd, idx in sb._ransac_planes(ds, 0.02, K=15, min_in=300, seed=0)
         if abs(n[2] / np.linalg.norm(n)) <= 0.5]

im = cv2.imread(path + "/color.png")
colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
for k, (n, dd, idx) in enumerate(cands[:5]):
    pts = ds[idx]
    quad, fill = window_quad(pts)
    pr, _ = cv2.projectPoints(quad.astype(np.float64), rvec, t, K, dist)
    pr = pr.reshape(-1, 2).astype(int)
    cv2.polylines(im, [pr], True, colors[k % 5], 3)
    print(f"#{k} |d|={abs(dd):.2f} n_z={n[2]:.2f} 内点{len(idx)} 窗内占比 {fill:.2f} 颜色{k}")

cv2.imwrite(f"/tmp/cands_{d}.png", im)
print(f"图: /tmp/cands_{d}.png  (0=黄 1=青 2=品红 3=绿 4=橙)")

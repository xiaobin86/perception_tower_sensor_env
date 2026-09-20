import numpy as np, sys
sys.path.insert(0, "/workspace")
import segment_board as sb
import open3d as o3d


def load(d):
    p = f"/workspace/turntable_output/20260918_{d}/merged.ply"
    with open(p) as f:
        for i, l in enumerate(f, 1):
            if l.strip() == "end_header":
                break
    return np.loadtxt(p, skiprows=i)[:, :3]


def window_stats(pts, L=0.84, W=0.60):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    u, v = (pts - c) @ e1, (pts - c) @ e2
    best = -1
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
            if row[j] > best:
                best = float(row[j])
    return best, 1 - best / len(pts)


for dname in ["072855", "073031"]:
    P = load(dname)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    ds = np.asarray(pcd.voxel_down_sample(0.01).points)
    raw = sb._ransac_planes(ds, 0.02, K=15, min_in=300, seed=0)
    cands = [(n / np.linalg.norm(n), dd, idx) for n, dd, idx in raw if abs(n[2] / np.linalg.norm(n)) <= 0.5]
    print(f"\n=== {dname}: {len(cands)} 竖直候选 ===")
    print(f"  |d|    n_z   内点    窗内    溢出")
    for n, dd, idx in cands[:8]:
        pts = ds[idx]
        cnt, ov = window_stats(pts)
        print(f"  {abs(dd):5.2f} {n[2]:5.2f} {len(idx):6d} {int(cnt):6d} {ov:6.2f}")
    # 合并近平行候选(法向点积>0.98 且 |Δd|<0.06), 再算溢出
    used = set()
    for i, (n1, d1, i1) in enumerate(cands):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j, (n2, d2, i2) in enumerate(cands):
            if j in used or j == i:
                continue
            if abs(float(n1 @ n2)) > 0.98 and abs(abs(d1) - abs(d2)) < 0.06:
                group.append(j)
                used.add(j)
        allidx = np.concatenate([cands[g][2] for g in group])
        pts = ds[allidx]
        cnt, ov = window_stats(pts)
        print(f"  合并{len(group)}个: 内点 {len(allidx):6d} 窗内 {int(cnt):6.0f} 溢出 {ov:.2f}")

import cv2, numpy as np, sys
def tv_photo(src, out, seed=0):
    rng = np.random.default_rng(seed)
    content = cv2.resize(cv2.imread(src), (1280, 720))
    H, W = 1080, 1440
    room = np.full((H, W, 3), (60, 70, 80), np.uint8)
    room[:] = room + rng.integers(0, 25, room.shape, dtype=np.uint8)
    cx, cy = W/2 + rng.uniform(-80, 80), H/2 + rng.uniform(-60, 60)
    tw, th = 900, 506
    j = lambda: rng.uniform(-45, 45)
    quad = np.float32([[cx-tw/2+j(), cy-th/2+j()], [cx+tw/2+j(), cy-th/2+j()], [cx+tw/2+j(), cy+th/2+j()], [cx-tw/2+j(), cy+th/2+j()]])
    bez = quad.copy(); c = quad.mean(0); bez = c + (bez - c) * 1.05
    cv2.fillConvexPoly(room, bez.astype(np.int32), (15, 15, 15))
    M = cv2.getPerspectiveTransform(np.float32([[0,0],[1279,0],[1279,719],[0,719]]), quad)
    warped = cv2.warpPerspective(content, M, (W, H))
    mask = cv2.warpPerspective(np.full((720,1280),255,np.uint8), M, (W, H))
    room[mask > 0] = warped[mask > 0]
    yy, xx = np.mgrid[0:H, 0:W]
    moire = (6*np.sin(xx*0.9 + yy*0.35)).astype(np.int16)[..., None]
    room = np.clip(room.astype(np.int16) + moire*(mask[...,None]>0), 0, 255).astype(np.uint8)
    cv2.imwrite(out, room); return quad
if __name__ == "__main__":
    print(tv_photo(sys.argv[1], sys.argv[2]).round().tolist())

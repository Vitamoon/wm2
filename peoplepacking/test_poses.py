from human3d import build_posed_human, get_bounding_box, packing_efficiency_3d, POSES_3D

for name in POSES_3D:
    m = build_posed_human(name)
    bb = get_bounding_box(m)
    eff = packing_efficiency_3d(m)
    print(f"{name:<30} BB: {bb[0]:.2f} x {bb[1]:.2f} x {bb[2]:.2f}  eff={eff:.0%}")

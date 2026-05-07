import re
from pathlib import Path
from PIL import Image

def frame_index(path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else -1

img_dir = Path(r"C:\Users\px2030\Code\PSD_opt\agggenerator\tests\crack_snapshots")
pngs = sorted(img_dir.glob("udp_crack*.png"), key=frame_index)

# img_dir = Path(r"C:\Users\px2030\Code\PSD_opt\agggenerator\tests\snapshot")
# pngs = sorted(img_dir.glob("material_mix*.png"), key=frame_index)


frames = [Image.open(p).convert("RGBA") for p in pngs]
out = img_dir / "udp_crack.gif"

frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=80,  
    loop=0,
    disposal=2,
)
帮我根据以下优化写个code plan，写在docs，之后我确认无误后将落实：


由于我的数据量级比较小，之前的方案很难在vison tracjectory上收敛，我决定优化一下

我将采用一个用于前馈重建的视觉基础模型VGGTOmega替换掉之前使用的PE模型

因为这类模型可以用来回归视频帧的相机参数，并且在大规模数据训练过，所以我觉得用在这个项目中非常合适

我将该模型的项目clone到了：third_party/vggt-omega，模型权重在：

使用该模型的一个基本的demo代码如下：

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "path/to/vggt_omega_1b_512.pt"
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]
```

你需要关注CameraHead的输入和结构，过渡到我的任务，此时encoder的输入就是VGGTOmega的CameraHead的输入，经过编码后的tokens我将作为action token，然后可以decode回轨迹


你可以大胆的改动，不用刻意的保留之前版本的模型和架构，因为它无法收敛，我肯定要改的
"""Generate a redistributable catalog and export its MobileNet browser encoder."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SITE = Path(__file__).parent / "product-site"
PALETTES = [
    ("Amber speaker", "#eab262", "#50342b"), ("Slate speaker", "#8196bb", "#24304a"),
    ("Spruce bottle", "#72a68d", "#234f42"), ("Lagoon bottle", "#65a5cc", "#264768"),
    ("Berry mug", "#cc7899", "#633957"), ("Sand mug", "#d8bf94", "#80664f"),
    ("Violet camera", "#9583c7", "#3f315c"), ("Cobalt camera", "#6b8bd5", "#273a70"),
]


def render(product, angle):
    label, color, dark = PALETTES[product]
    image = Image.new("RGB", (224,224), "#eaf0f7")
    draw = ImageDraw.Draw(image)
    x = 112 + (angle-1)*9
    width = 48-angle*5
    draw.ellipse((x-68,172,x+68,188),fill="#cbd6e1")
    if "speaker" in label:
        draw.rounded_rectangle((x-width,47,x+width,176),15,fill=color,outline=dark,width=4)
        for y,r in ((87,18),(135,26)):
            draw.ellipse((x-r,y-r,x+r,y+r),fill=dark)
            draw.ellipse((x-r+7,y-r+7,x+r-7,y+r-7),fill=color)
    elif "bottle" in label:
        draw.rounded_rectangle((x-width+14,70,x+width-14,175),18,fill=color,outline=dark,width=4)
        draw.rectangle((x-13,48,x+13,74),fill=dark)
        draw.rectangle((x-width+17,108,x+width-17,133),fill="#f3eddc")
    elif "mug" in label:
        draw.ellipse((x+19,85,x+width+25,146),outline=dark,width=9)
        draw.rounded_rectangle((x-width,71,x+width,170),14,fill=color,outline=dark,width=4)
        draw.ellipse((x-width+4,66,x+width-4,92),fill=dark)
    else:
        draw.rounded_rectangle((x-width-10,78,x+width+10,154),10,fill=color,outline=dark,width=4)
        draw.rectangle((x-14,62,x+16,82),fill=dark)
        draw.ellipse((x-27,88,x+27,142),fill=dark,outline="#f3eddc",width=3)
    return image


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

    def forward(self, inputs):
        return self.model.avgpool(self.model.features(inputs)).flatten(1)


def main():
    torch.set_num_threads(2)
    SITE.mkdir(exist_ok=True)
    model = Encoder().eval()
    catalog = []
    queries = []
    with torch.inference_mode():
        for key, (label, _, _) in enumerate(PALETTES):
            for angle in range(4):
                item = render(key, angle)
                file = f"object-{key}-{angle}.png"
                item.save(SITE / file)
                if angle == 3:
                    queries.append({"id":key, "label":label, "image":file})
                    continue
                catalog.append((key,label,file,item))
    from torchvision.transforms.functional import to_tensor, normalize
    with torch.inference_mode():
        vectors = torch.nn.functional.normalize(model(torch.stack([
            normalize(to_tensor(item), [.485,.456,.406], [.229,.224,.225]) for _,_,_,item in catalog
        ]))).numpy()
    rows = [{"id":key,"label":label,"image":file,"vector":np.round(vector,7).tolist()}
            for (key,label,file,_),vector in zip(catalog,vectors)]
    (SITE / "catalog.json").write_text(json.dumps({"gallery":rows,"queries":queries},separators=(",",":")))
    torch.onnx.export(model, torch.zeros(1,3,224,224), SITE / "mobilenet.onnx",
                      input_names=["pixels"], output_names=["embedding"], opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()

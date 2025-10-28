from PIL import Image
from PIL import ImageDraw

class GoldenRectangle():
    
    def __init__(self):
        self.line_color = (255, 0, 0)
        self.line_width = 4

    def process(
        self,
        data_input: Image.Image,
        **kwargs,
    ) -> Image.Image:
        return self.apply_on_image(data_input)

    def apply_on_image(self, image) -> Image.Image:
        draw = ImageDraw.Draw(image)

        width, height = image.size

        phi = (1 + 5**0.5) / 2

        vertical_line_1 = int(width / phi)
        vertical_line_2 = width - vertical_line_1

        horizontal_line_1 = int(height / phi)
        horizontal_line_2 = height - horizontal_line_1

        draw.line(
            [(vertical_line_1, 0), (vertical_line_1, height)],
            fill=self.line_color,
            width=self.line_width,
        )
        draw.line(
            [(vertical_line_2, 0), (vertical_line_2, height)],
            fill=self.line_color,
            width=self.line_width,
        )

        draw.line(
            [(0, horizontal_line_1), (width, horizontal_line_1)],
            fill=self.line_color,
            width=self.line_width,
        )
        draw.line(
            [(0, horizontal_line_2), (width, horizontal_line_2)],
            fill=self.line_color,
            width=self.line_width,
        )

        return image
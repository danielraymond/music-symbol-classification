import itertools

import numpy as np
import torch
from numpy import random
from PIL import Image


class SaltAndPepperNoise:
    def __init__(self, salt_prob=0.02, pepper_prob=0.02):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, img):
        device = img.device if isinstance(img, torch.Tensor) else "cpu"
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        noisy = np.copy(img)
        total_pixels = img.shape[1] * img.shape[2]
        num_salt = int(self.salt_prob * total_pixels)
        num_pepper = int(self.pepper_prob * total_pixels)

        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[1:]]
        noisy[:, coords[0], coords[1]] = 1

        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[1:]]
        noisy[:, coords[0], coords[1]] = 0

        return torch.tensor(noisy).to(device)


class ElasticDistortion:
    def __init__(self, grid, magnitude, min_sep):
        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, img):
        device = img.device if isinstance(img, torch.Tensor) else "cpu"
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        w, h = img.shape[2], img.shape[1]

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(w / float(horizontal_tiles))
        height_of_square = int(h / float(vertical_tiles))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [
            [(0, 0) for _ in range(horizontal_tiles)] for _ in range(vertical_tiles)
        ]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (
                    horizontal_tiles - 1
                ):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )
                else:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )

                sm_h = (
                    min(
                        self.xmagnitude,
                        width_of_square
                        - (
                            self.min_h_sep
                            + shift[vertical_tile][horizontal_tile - 1][0]
                        ),
                    )
                    if horizontal_tile > 0
                    else self.xmagnitude
                )
                sm_v = (
                    min(
                        self.ymagnitude,
                        height_of_square
                        - (
                            self.min_v_sep
                            + shift[vertical_tile - 1][horizontal_tile][1]
                        ),
                    )
                    if vertical_tile > 0
                    else self.ymagnitude
                )

                sm_h = max(sm_h, 1)
                sm_v = max(sm_v, 1)

                dx = random.randint(-sm_h, sm_h)
                dy = random.randint(-sm_v, sm_v)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = [
            (horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)
        ]
        last_row = range(
            (horizontal_tiles * vertical_tiles) - horizontal_tiles,
            horizontal_tiles * vertical_tiles,
        )

        polygons = []
        for i, (x1, y1, x2, y2) in enumerate(dimensions):
            dx, dy = shift[i]
            x3 = x2 + dx
            y3 = y2 + dy
            x4 = x1 + dx
            y4 = y1 + dy
            polygons.append([x1, y1, x2, y2, x3, y3, x4, y4])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append(
                    [i, i + 1, i + vertical_tiles, i + 1 + vertical_tiles]
                )

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = [[dimensions[i], polygons[i]] for i in range(len(dimensions))]

        img = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        img = img.transform(
            img.size, Image.MESH, generated_mesh, resample=Image.BICUBIC
        )
        img = np.array(img).astype(np.float32) / 255.0

        return torch.tensor(img).permute(2, 0, 1).to(device)


class Fade:
    def __init__(self, fade_factor=0.5):
        self.fade_factor = fade_factor

    def __call__(self, img):
        return img * self.fade_factor

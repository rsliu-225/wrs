import os
import random

from svgpathtools import svg2paths

import utils.drawpath_utils as du


class KenjiSvgSampler:
    def sample_stroke_points(self, svg_path, max_num_samples=50):
        strokes_sampled_points = []
        strokes, _ = svg2paths(svg_path)

        stroke_lengths = [s.length() for s in strokes]
        max_stroke_len = max(stroke_lengths)
        stroke_num_samples = [int((length / max_stroke_len) * max_num_samples) for length in stroke_lengths]
        print(f"Sample points per stroke: {stroke_num_samples}")

        for ind, stroke in enumerate(strokes):
            stroke_sampled_points = []
            num_samples = stroke_num_samples[ind]
            for i in range(num_samples):
                complex_point = stroke.point(i / (num_samples - 1))
                stroke_sampled_points.append((complex_point.real, -complex_point.imag))
            strokes_sampled_points.append(stroke_sampled_points)

        return strokes_sampled_points


if __name__ == '__main__':
    sampler = KenjiSvgSampler()
    # svg_f_name = random.choice(os.listdir("./kanji"))
    svg_f_name = "A.svg"

    pkl_f_name = str(svg_f_name).split(".svg")[0]+".pkl"

    svg_path = "./kanji/" + svg_f_name
    pointlist_ms = sampler.sample_stroke_points(svg_path, 30)
    for stroke in pointlist_ms:
        du.draw_by_plist(stroke)
    du.plot_ms(pointlist_ms)
    du.dump_drawpath(pointlist_ms, pkl_f_name)

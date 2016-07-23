// This file is a part of WCPSP_Stereo. License is MIT.
// Copyright (c), Qingqing Yang

#include "stereo_lib.h"
#include <stdlib.h>
#include <stack>

namespace cvlab {

void speckle_filter(Image<int>& input_image,
                    int max_speckle_size,
                    int max_diff) {

    int width = input_image.width();
    int height = input_image.height();
    int* image = input_image.data();

	std::vector<int> labels(width*height, 0);
	std::vector<bool> region_types(1);
	region_types[0] = false;

	int current_label_index = 0;

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			int pixel_index = width*y + x;
			if (image[width*y + x] != 0) {
				if (labels[pixel_index] > 0) {
					if (region_types[labels[pixel_index]]) {
						image[width*y + x] = 0;
					}
				} else {
					std::stack<int> wavefront_indices;
					wavefront_indices.push(pixel_index);
					++current_label_index;
					region_types.push_back(false);
					int region_pixel_tobal = 0;
					labels[pixel_index] = current_label_index;

					while (!wavefront_indices.empty()) {
						int current_pixel_index = wavefront_indices.top();
						wavefront_indices.pop();
						int current_x = current_pixel_index%width;
						int current_y = current_pixel_index/width;
						++region_pixel_tobal;
						uint16_t pixel_value = image[width*current_y + current_x];

						if (current_x < width-1 && labels[current_pixel_index+1] == 0 &&
                            image[width*current_y + current_x + 1] != 0 &&
                            std::abs(pixel_value-image[width*current_y+current_x+1]) <= max_diff) {
							labels[current_pixel_index+1] = current_label_index;
							wavefront_indices.push(current_pixel_index + 1);
						}

						if (current_x > 0 && labels[current_pixel_index-1] == 0 &&
                            image[width*current_y+current_x-1] != 0 &&
                            std::abs(pixel_value - image[width*current_y+current_x-1]) <= max_diff) {
							labels[current_pixel_index-1] = current_label_index;
							wavefront_indices.push(current_pixel_index - 1);
						}

						if (current_y < height-1 && labels[current_pixel_index+width] == 0 &&
                            image[width*(current_y+1)+current_x] != 0 &&
                            std::abs(pixel_value-image[width*(current_y+1)+current_x]) <= max_diff) {
							labels[current_pixel_index+width] = current_label_index;
							wavefront_indices.push(current_pixel_index + width);
						}

						if (current_y > 0 && labels[current_pixel_index-width] == 0 &&
                            image[width*(current_y-1)+current_x] != 0 &&
                            std::abs(pixel_value-image[width*(current_y-1)+current_x]) <= max_diff) {
							labels[current_pixel_index - width] = current_label_index;
							wavefront_indices.push(current_pixel_index - width);
						}
					}

					if (region_pixel_tobal <= max_speckle_size) {
						region_types[current_label_index] = true;
						image[width*y + x] = 0;
					}
				}
			}
		}
	}
}

}  // namespace cvlab

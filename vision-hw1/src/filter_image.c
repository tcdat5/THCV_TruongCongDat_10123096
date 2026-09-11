#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "image.h"

#define TWO_PI 6.28318530717958647692f

void l1_normalize(image im)
{
    int count = im.w * im.h * im.c;
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += im.data[i];
    if (sum == 0.0f) return;
    for (int i = 0; i < count; ++i) im.data[i] /= sum;
}

image make_box_filter(int w)
{
    assert(w > 0);
    image filter = make_image(w, w, 1);
    for (int i = 0; i < w * w; ++i) filter.data[i] = 1.0f;
    l1_normalize(filter);
    return filter;
}

image convolve_image(image im, image filter, int preserve)
{
    assert(filter.c == 1 || filter.c == im.c);
    int output_channels = preserve ? im.c : 1;
    image result = make_image(im.w, im.h, output_channels);
    int center_x = filter.w / 2;
    int center_y = filter.h / 2;

    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            for (int out_c = 0; out_c < output_channels; ++out_c) {
                float sum = 0.0f;
                int first_channel = preserve ? out_c : 0;
                int last_channel = preserve ? out_c + 1 : im.c;

                for (int c = first_channel; c < last_channel; ++c) {
                    int filter_channel = filter.c == 1 ? 0 : c;
                    for (int fy = 0; fy < filter.h; ++fy) {
                        for (int fx = 0; fx < filter.w; ++fx) {
                            float pixel = get_pixel(im,
                                                    x + fx - center_x,
                                                    y + fy - center_y, c);
                            float weight = get_pixel(filter, fx, fy,
                                                     filter_channel);
                            sum += pixel * weight;
                        }
                    }
                }
                set_pixel(result, x, y, out_c, sum);
            }
        }
    }
    return result;
}

static image make_3x3_filter(const float values[9])
{
    image filter = make_image(3, 3, 1);
    memcpy(filter.data, values, 9 * sizeof(float));
    return filter;
}

image make_highpass_filter()
{
    const float values[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };
    return make_3x3_filter(values);
}

image make_sharpen_filter()
{
    const float values[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };
    return make_3x3_filter(values);
}

image make_emboss_filter()
{
    const float values[9] = {
        -2, -1, 0,
        -1,  1, 1,
         0,  1, 2
    };
    return make_3x3_filter(values);
}

// Question 2.2.1: Which filters should preserve channels, and why?
// Answer: Preserve channels for sharpen and emboss so each RGB channel is
// filtered independently and the output remains colored. For high-pass edge
// visualization, do not preserve: summing the channel responses produces a
// single-channel edge image.

// Question 2.2.2: Is post-processing needed, and why?
// Answer: Clamp the outputs before saving. Negative kernel weights can produce
// values below 0, while sharpen and emboss may also produce values above 1;
// neither range can be represented correctly by the image writer.

image make_gaussian_filter(float sigma)
{
    assert(sigma > 0.0f);
    int size = (int)ceilf(6.0f * sigma);
    if (size % 2 == 0) ++size;

    image filter = make_image(size, size, 1);
    int center = size / 2;
    float variance = sigma * sigma;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dx = (float)(x - center);
            float dy = (float)(y - center);
            float exponent = -(dx * dx + dy * dy) / (2.0f * variance);
            set_pixel(filter, x, y, 0,
                      expf(exponent) / (TWO_PI * variance));
        }
    }
    l1_normalize(filter);
    return filter;
}

static void assert_same_shape(image a, image b)
{
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
}

image add_image(image a, image b)
{
    assert_same_shape(a, b);
    image result = make_image(a.w, a.h, a.c);
    int count = a.w * a.h * a.c;
    for (int i = 0; i < count; ++i) result.data[i] = a.data[i] + b.data[i];
    return result;
}

image sub_image(image a, image b)
{
    assert_same_shape(a, b);
    image result = make_image(a.w, a.h, a.c);
    int count = a.w * a.h * a.c;
    for (int i = 0; i < count; ++i) result.data[i] = a.data[i] - b.data[i];
    return result;
}

image make_gx_filter()
{
    const float values[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    return make_3x3_filter(values);
}

image make_gy_filter()
{
    const float values[9] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    return make_3x3_filter(values);
}

void feature_normalize(image im)
{
    int count = im.w * im.h * im.c;
    if (count == 0) return;

    float minimum = FLT_MAX;
    float maximum = -FLT_MAX;
    for (int i = 0; i < count; ++i) {
        if (im.data[i] < minimum) minimum = im.data[i];
        if (im.data[i] > maximum) maximum = im.data[i];
    }

    float range = maximum - minimum;
    if (range == 0.0f) {
        memset(im.data, 0, (size_t)count * sizeof(float));
        return;
    }
    for (int i = 0; i < count; ++i) {
        im.data[i] = (im.data[i] - minimum) / range;
    }
}

image *sobel_image(image im)
{
    image gx_filter = make_gx_filter();
    image gy_filter = make_gy_filter();
    image gx = convolve_image(im, gx_filter, 0);
    image gy = convolve_image(im, gy_filter, 0);
    image magnitude = make_image(im.w, im.h, 1);
    image direction = make_image(im.w, im.h, 1);

    int count = im.w * im.h;
    for (int i = 0; i < count; ++i) {
        magnitude.data[i] = hypotf(gx.data[i], gy.data[i]);
        direction.data[i] = atan2f(gy.data[i], gx.data[i]);
    }

    image *result = calloc(2, sizeof(image));
    assert(result != 0);
    result[0] = magnitude;
    result[1] = direction;

    free_image(gx_filter);
    free_image(gy_filter);
    free_image(gx);
    free_image(gy);
    return result;
}

image colorize_sobel(image im)
{
    image *sobel = sobel_image(im);
    image magnitude = sobel[0];
    image direction = sobel[1];
    feature_normalize(magnitude);
    feature_normalize(direction);

    image output = make_image(im.w, im.h, 3);
    int count = im.w * im.h;
    for (int i = 0; i < count; ++i) {
        output.data[i] = direction.data[i];
        output.data[count + i] = magnitude.data[i];
        output.data[2 * count + i] = magnitude.data[i];
    }
    hsv_to_rgb(output);

    free_image(magnitude);
    free_image(direction);
    free(sobel);
    return output;
}

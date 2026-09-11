#include <assert.h>
#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
    return get_pixel(im, (int)roundf(x), (int)roundf(y), c);
}

static float source_coordinate(int destination, int source_size,
                               int destination_size)
{
    return ((destination + 0.5f) * source_size / destination_size) - 0.5f;
}

image nn_resize(image im, int w, int h)
{
    assert(w > 0 && h > 0);
    image resized = make_image(w, h, im.c);

    for (int y = 0; y < h; ++y) {
        float source_y = source_coordinate(y, im.h, h);
        for (int x = 0; x < w; ++x) {
            float source_x = source_coordinate(x, im.w, w);
            for (int c = 0; c < im.c; ++c) {
                set_pixel(resized, x, y, c,
                          nn_interpolate(im, source_x, source_y, c));
            }
        }
    }
    return resized;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    int left = (int)floorf(x);
    int top = (int)floorf(y);
    float dx = x - left;
    float dy = y - top;

    float top_value = (1.0f - dx) * get_pixel(im, left, top, c)
                    + dx * get_pixel(im, left + 1, top, c);
    float bottom_value = (1.0f - dx) * get_pixel(im, left, top + 1, c)
                       + dx * get_pixel(im, left + 1, top + 1, c);
    return (1.0f - dy) * top_value + dy * bottom_value;
}

image bilinear_resize(image im, int w, int h)
{
    assert(w > 0 && h > 0);
    image resized = make_image(w, h, im.c);

    for (int y = 0; y < h; ++y) {
        float source_y = source_coordinate(y, im.h, h);
        for (int x = 0; x < w; ++x) {
            float source_x = source_coordinate(x, im.w, w);
            for (int c = 0; c < im.c; ++c) {
                set_pixel(resized, x, y, c,
                          bilinear_interpolate(im, source_x, source_y, c));
            }
        }
    }
    return resized;
}

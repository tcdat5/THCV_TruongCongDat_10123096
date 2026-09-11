#include <assert.h>
#include <math.h>
#include <string.h>
#include "image.h"

static int clamp_coordinate(int value, int upper_bound)
{
    if (value < 0) return 0;
    if (value >= upper_bound) return upper_bound - 1;
    return value;
}

float get_pixel(image im, int x, int y, int c)
{
    assert(im.data != 0 && im.w > 0 && im.h > 0 && im.c > 0);
    x = clamp_coordinate(x, im.w);
    y = clamp_coordinate(y, im.h);
    c = clamp_coordinate(c, im.c);
    return im.data[c * im.w * im.h + y * im.w + x];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    if (!im.data || x < 0 || x >= im.w || y < 0 || y >= im.h ||
        c < 0 || c >= im.c) {
        return;
    }
    im.data[c * im.w * im.h + y * im.w + x] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    size_t count = (size_t)im.w * im.h * im.c;
    memcpy(copy.data, im.data, count * sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);

    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            float value = 0.299f * get_pixel(im, x, y, 0)
                        + 0.587f * get_pixel(im, x, y, 1)
                        + 0.114f * get_pixel(im, x, y, 2);
            set_pixel(gray, x, y, 0, value);
        }
    }
    return gray;
}

void shift_image(image im, int c, float v)
{
    if (c < 0 || c >= im.c) return;
    int offset = c * im.w * im.h;
    int channel_size = im.w * im.h;
    for (int i = 0; i < channel_size; ++i) im.data[offset + i] += v;
}

void scale_image(image im, int c, float v)
{
    if (c < 0 || c >= im.c) return;
    int offset = c * im.w * im.h;
    int channel_size = im.w * im.h;
    for (int i = 0; i < channel_size; ++i) im.data[offset + i] *= v;
}

void clamp_image(image im)
{
    int count = im.w * im.h * im.c;
    for (int i = 0; i < count; ++i) {
        if (im.data[i] < 0.0f) im.data[i] = 0.0f;
        else if (im.data[i] > 1.0f) im.data[i] = 1.0f;
    }
}

static float maximum3(float a, float b, float c)
{
    return fmaxf(a, fmaxf(b, c));
}

static float minimum3(float a, float b, float c)
{
    return fminf(a, fminf(b, c));
}

void rgb_to_hsv(image im)
{
    assert(im.c == 3);
    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            float r = get_pixel(im, x, y, 0);
            float g = get_pixel(im, x, y, 1);
            float b = get_pixel(im, x, y, 2);
            float v = maximum3(r, g, b);
            float chroma = v - minimum3(r, g, b);
            float s = v == 0.0f ? 0.0f : chroma / v;
            float h = 0.0f;

            if (chroma != 0.0f) {
                if (v == r) h = (g - b) / chroma;
                else if (v == g) h = (b - r) / chroma + 2.0f;
                else h = (r - g) / chroma + 4.0f;
                h /= 6.0f;
                if (h < 0.0f) h += 1.0f;
            }

            set_pixel(im, x, y, 0, h);
            set_pixel(im, x, y, 1, s);
            set_pixel(im, x, y, 2, v);
        }
    }
}

void hsv_to_rgb(image im)
{
    assert(im.c == 3);
    for (int y = 0; y < im.h; ++y) {
        for (int x = 0; x < im.w; ++x) {
            float h = get_pixel(im, x, y, 0);
            float s = get_pixel(im, x, y, 1);
            float v = get_pixel(im, x, y, 2);
            float chroma = v * s;
            float h6 = 6.0f * (h - floorf(h));
            float second = chroma * (1.0f - fabsf(fmodf(h6, 2.0f) - 1.0f));
            float m = v - chroma;
            float r1, g1, b1;

            if (h6 < 1.0f) {
                r1 = chroma; g1 = second; b1 = 0.0f;
            } else if (h6 < 2.0f) {
                r1 = second; g1 = chroma; b1 = 0.0f;
            } else if (h6 < 3.0f) {
                r1 = 0.0f; g1 = chroma; b1 = second;
            } else if (h6 < 4.0f) {
                r1 = 0.0f; g1 = second; b1 = chroma;
            } else if (h6 < 5.0f) {
                r1 = second; g1 = 0.0f; b1 = chroma;
            } else {
                r1 = chroma; g1 = 0.0f; b1 = second;
            }

            set_pixel(im, x, y, 0, r1 + m);
            set_pixel(im, x, y, 1, g1 + m);
            set_pixel(im, x, y, 2, b1 + m);
        }
    }
}

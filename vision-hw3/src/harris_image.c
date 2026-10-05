#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#include "matrix.h"
#include <time.h>

// Frees an array of descriptors.
// descriptor *d: the array.
// int n: number of elements in array.
void free_descriptors(descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        free(d[i].data);
    }
    free(d);
}

// Create a feature descriptor for an index in an image.
// image im: source image.
// int i: index in image for the pixel we want to describe.
// returns: descriptor for that index.
descriptor describe_index(image im, int i)
{
    int w = 5;
    descriptor d;
    d.p.x = i%im.w;
    d.p.y = i/im.w;
    d.data = calloc(w*w*im.c, sizeof(float));
    d.n = w*w*im.c;
    int c, dx, dy;
    int count = 0;
    // If you want you can experiment with other descriptors
    // This subtracts the central value from neighbors
    // to compensate some for exposure/lighting changes.
    for(c = 0; c < im.c; ++c){
        float cval = im.data[c*im.w*im.h + i];
        for(dx = -w/2; dx < (w+1)/2; ++dx){
            for(dy = -w/2; dy < (w+1)/2; ++dy){
                float val = get_pixel(im, i%im.w+dx, i/im.w+dy, c);
                d.data[count++] = cval - val;
            }
        }
    }
    return d;
}

// Marks the spot of a point in an image.
// image im: image to mark.
// ponit p: spot to mark in the image.
void mark_spot(image im, point p)
{
    int x = p.x;
    int y = p.y;
    int i;
    for(i = -9; i < 10; ++i){
        set_pixel(im, x+i, y, 0, 1);
        set_pixel(im, x, y+i, 0, 1);
        set_pixel(im, x+i, y, 1, 0);
        set_pixel(im, x, y+i, 1, 0);
        set_pixel(im, x+i, y, 2, 1);
        set_pixel(im, x, y+i, 2, 1);
    }
}

// Marks corners denoted by an array of descriptors.
// image im: image to mark.
// descriptor *d: corners in the image.
// int n: number of descriptors to mark.
void mark_corners(image im, descriptor *d, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        mark_spot(im, d[i].p);
    }
}

// Creates a 1d Gaussian filter.
// float sigma: standard deviation of Gaussian.
// returns: single row image of the filter.
image make_1d_gaussian(float sigma)
{
    assert(sigma > 0);
    int f = (int)ceilf(6*sigma);
    if(!(f & 1)) ++f;

    image filter = make_image(f, 1, 1);
    float denom = 2*sigma*sigma;
    int center = f/2;
    for(int i = 0; i < f; ++i){
        float x = i-center;
        filter.data[i] = expf(-(x*x)/denom);
    }
    l1_normalize(filter);
    return filter;
}

// Smooths an image using separable Gaussian filter.
// image im: image to smooth.
// float sigma: std dev. for Gaussian.
// returns: smoothed image.
image smooth_image(image im, float sigma)
{
    image filter = make_1d_gaussian(sigma);
    image mid = convolve_image(im, filter, 1);
    filter.h = filter.w;
    filter.w = 1;
    image filtered_image = convolve_image(mid, filter, 1);
    free_image(mid);
    free_image(filter);
    return filtered_image;
}
// Calculate the structure matrix of an image.
// image im: the input image.
// float sigma: std dev. to use for weighted sum.
// returns: structure matrix. 1st channel is Ix^2, 2nd channel is Iy^2,
//          third channel is IxIy.
image structure_matrix(image im, float sigma)
{
    image S = make_image(im.w, im.h, 3);
    image gx = make_gx_filter(), gy = make_gy_filter();
    image Ix = convolve_image(im, gx, 0), Iy = convolve_image(im, gy, 0);
    for(int i=0; i<im.w; ++i){
        for(int j=0; j<im.h; ++j){
            float ix = get_pixel(Ix, i, j, 0);
            float iy = get_pixel(Iy, i, j, 0);
            set_pixel(S, i, j, 0, ix*ix);
            set_pixel(S, i, j, 1, iy*iy);
            set_pixel(S, i, j, 2, ix*iy);
        }
    }
    image weighted_S = smooth_image(S, sigma);
    free_image(gx);
    free_image(gy);
    free_image(Ix);
    free_image(Iy);
    free_image(S);
    return weighted_S;
}

// Estimate the cornerness of each pixel given a structure matrix S.
// image S: structure matrix for an image.
// returns: a response map of cornerness calculations.
image cornerness_response(image S)
{
    image R = make_image(S.w, S.h, 1);
    for(int i=0; i<S.w; ++i){
        for(int j=0; j<S.h; ++j){
            float Ixx = get_pixel(S, i, j, 0);
            float Iyy = get_pixel(S, i, j, 1);
            float Ixy = get_pixel(S, i, j, 2);
            float trace = Ixx + Iyy;
            set_pixel(R, i, j, 0, Ixx*Iyy - Ixy*Ixy - .06f*trace*trace);
        }
    }
    return R;
}

// Perform non-max supression on an image of feature responses.
// image im: 1-channel image of feature responses.
// int w: distance to look for larger responses.
// returns: image with only local-maxima responses within w pixels.
image nms_image(image im, int w)
{
    assert(im.c == 1);
    assert(w >= 0);
    image r = copy_image(im);
    for(int i=0; i<r.w; ++i){
        for(int j=0; j<r.h; ++j){
            float value = get_pixel(im, i, j, 0);
            int suppressed = 0;
            for(int y = MAX(0, j-w); y <= MIN(im.h-1, j+w) && !suppressed; ++y){
                for(int x = MAX(0, i-w); x <= MIN(im.w-1, i+w); ++x){
                    if(get_pixel(im, x, y, 0) > value){
                        suppressed = 1;
                        break;
                    }
                }
            }
            if(suppressed) set_pixel(r, i, j, 0, -999999);
        }
    }
    return r;
}

// Perform harris corner detection and extract features from the corners.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
// int *n: pointer to number of corners detected, should fill in.
// returns: array of descriptors of the corners in the image.
descriptor *harris_corner_detector(image im, float sigma, float thresh, int nms, int *n)
{
    // Calculate structure matrix
    image S = structure_matrix(im, sigma);

    // Estimate cornerness
    image R = cornerness_response(S);

    // Run NMS on the responses
    image Rnms = nms_image(R, nms);


    int count = 0;
    for(int i=0; i<(Rnms.w*Rnms.h); ++i){
        if(*(Rnms.data + i)>thresh)
            ++count;
    }
    
    *n = count;
    descriptor *d = calloc(count, sizeof(descriptor));
    int idx = 0;
    for(int i = 0; i<(Rnms.w*Rnms.h); ++i){
        if(*(Rnms.data + i) > thresh)
            d[idx++] = describe_index(im, i);
    }
    free_image(S);
    free_image(R);
    free_image(Rnms);
    return d;
}

// Find and draw corners on an image.
// image im: input image.
// float sigma: std. dev for harris.
// float thresh: threshold for cornerness.
// int nms: distance to look for local-maxes in response map.
void detect_and_draw_corners(image im, float sigma, float thresh, int nms)
{
    int n = 0;
    descriptor *d = harris_corner_detector(im, sigma, thresh, nms, &n);
    mark_corners(im, d, n);
    free_descriptors(d, n);
}

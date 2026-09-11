from uwimg import *


# im = load_image("data/dogsmall.jpg")
# a = nn_resize(im, im.w*4, im.h*4)
# save_image(a, "dog4x-nn")



# a = bilinear_resize(im, im.w*4, im.h*4)
# save_image(a, "dog4x-bl")



# im = load_image("data/dog.jpg")
# a = nn_resize(im, im.w//7, im.h//7)
# save_image(a, "dog7th-bl")



# im = load_image("data/dog.jpg")
# f = make_box_filter(7)
# blur = convolve_image(im, f, 1)
# save_image(blur, "dog-box7")



# im = load_image("data/dog.jpg")
# f = make_box_filter(7)
# blur = convolve_image(im, f, 1)
# thumb = nn_resize(blur, blur.w//7, blur.h//7)
# save_image(thumb, "dogthumb")



# im = load_image("data/dog.jpg")
# f = make_highpass_filter()
# edge = convolve_image(im, f, 1)
# clamp_image(edge)
# save_image(edge, "dog-highpass")



# im = load_image("data/dog.jpg")
# f = make_sharpen_filter()
# sharp = convolve_image(im, f, 1)
# clamp_image(sharp)
# save_image(sharp, "dog-sharpen")



# im = load_image("data/dog.jpg")
# f = make_emboss_filter()
# emb = convolve_image(im, f, 1)
# clamp_image(emb)
# save_image(emb, "dog-emboss")



# im = load_image("data/dog.jpg")
# f = make_gaussian_filter(2)
# blur = convolve_image(im, f, 1)
# save_image(blur, "dog-gauss2")



# im = load_image("data/dog.jpg")
# f = make_gaussian_filter(2)
# lfreq = convolve_image(im, f, 1)
# hfreq = im - lfreq
# reconstruct = lfreq + hfreq
# save_image(lfreq, "low-frequency")
# save_image(hfreq, "high-frequency")
# save_image(reconstruct, "reconstruct")



im1 = load_image("data/ron.jpg")
highpass_filter = make_gaussian_filter(3)
lf1 = convolve_image(im1, highpass_filter, 1)
hf1 = im1 - lf1

im2 = load_image("data/dumbledore.jpg")
lowpass_filter = make_gaussian_filter(3)
lf2 = convolve_image(im2, lowpass_filter, 1)

# Keep negative high-frequency values until after the two images are combined.
final = hf1 + lf2
clamp_image(final)
save_image(final, "ronbledore")

free_image(im1)
free_image(im2)
free_image(highpass_filter)
free_image(lowpass_filter)
free_image(lf1)
free_image(hf1)
free_image(lf2)
free_image(final)



# im = load_image("data/dog.jpg")
# res = sobel_image(im)
# mag = res[0]
# feature_normalize(mag)
# save_image(mag, "magnitude")



im = load_image("data/dog.jpg")
f = make_gaussian_filter(3)
res = convolve_image(im, f, 1)
res = colorize_sobel(res)
clamp_image(res)
save_image(res, "sobel")

free_image(im)
free_image(f)
free_image(res)

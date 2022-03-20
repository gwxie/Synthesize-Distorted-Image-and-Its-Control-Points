import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.spatial.qhull as qhull
import math
import cv2

class BasePerturbed(object):
	# d = np.abs(sk_normalize(d, norm='l2'))

	def get_normalize(self, d):
		E = np.mean(d)
		std = np.std(d)
		d = (d-E)/std
		# d = preprocessing.normalize(d, norm='l2')
		return d

	def get_0_1_d(self, d, new_max=1, new_min=0):
		d_min = np.min(d)
		d_max = np.max(d)
		d = ((d-d_min)/(d_max-d_min))*(new_max-new_min)+new_min
		return d

	def draw_distance_hotmap(self, distance_vertex_line):

		plt.matshow(distance_vertex_line, cmap='autumn')
		plt.colorbar()
		plt.show()

	def get_pixel(self, p, origin_img):
		try:
			return origin_img[p[0], p[1]]
		except:
			# print('out !')
			return np.array([257, 257, 257])

	def nearest_neighbor_interpolation(self, xy, new_origin_img):
		# xy = np.around(xy_).astype(np.int)
		origin_pixel = self.get_pixel([xy[0], xy[1]], new_origin_img)
		if (origin_pixel == 256).all():
			return origin_pixel, False
		return origin_pixel, True

	def bilinear_interpolation(self, xy_, new_origin_img):
		xy_int = [int(xy_[0]), int(xy_[1])]
		xy_decimal = [round(xy_[0] - xy_int[0], 5), round(xy_[1] - xy_int[1], 5)]
		x0_y0 = (1 - xy_decimal[0]) * (1 - xy_decimal[1]) * self.get_pixel([xy_int[0], xy_int[1]], new_origin_img)

		x0_y1 = (1 - xy_decimal[0]) * (xy_decimal[1]) * self.get_pixel([xy_int[0], xy_int[1] + 1], new_origin_img)

		x1_y0 = (xy_decimal[0]) * (1 - xy_decimal[1]) * self.get_pixel([xy_int[0] + 1, xy_int[1]], new_origin_img)

		x1_y1 = (xy_decimal[0]) * (xy_decimal[1]) * self.get_pixel([xy_int[0] + 1, xy_int[1] + 1], new_origin_img)

		return x0_y0, x0_y1, x1_y0, x1_y1

	def get_coor(self, p, origin_label):
		try:
			return origin_label[p[0], p[1]]
		except:
			# print('out !')
			return np.array([0, 0])

	def bilinear_interpolation_coordinate_v4(self, xy_, new_origin_img):

		xy_int = [int(xy_[0]), int(xy_[1])]
		xy_decimal = [round(xy_[0] - xy_int[0], 5), round(xy_[1] - xy_int[1], 5)]
		x_y_i = 0
		x0, x1, x2, x3 = 0, 0, 0, 0
		y0, y1, y2, y3 = 0, 0, 0, 0
		x0_y0 = self.get_coor(np.array([xy_int[0], xy_int[1]]), new_origin_img)
		x0_y1 = self.get_coor(np.array([xy_int[0], xy_int[1]+1]), new_origin_img)
		x1_y0 = self.get_coor(np.array([xy_int[0]+1, xy_int[1]]), new_origin_img)
		x1_y1 = self.get_coor(np.array([xy_int[0]+1, xy_int[1]+1]), new_origin_img)

		if x0_y0[0] != 0:
			x0 = (1 - xy_decimal[0])
		if x0_y1[0] != 0:
			x1 = (1 - xy_decimal[0])
		if x1_y0[0] != 0:
			x2 = (xy_decimal[0])
		if x1_y1[0] != 0:
			x3 = (xy_decimal[0])

		if x0_y0[1] != 0:
			y0 = (1 - xy_decimal[1])
		if x0_y1[1] != 0:
			y1 = (xy_decimal[1])
		if x1_y0[1] != 0:
			y2 = (1 - xy_decimal[1])
		if x1_y1[1] != 0:
			y3 = (xy_decimal[1])

		x_ = x0+x1+x2+x3
		if x_ == 0:
			x = 0
		else:
			x = x0/x_*x0_y0[0]+x1/x_*x0_y1[0]+x2/x_*x1_y0[0]+x3/x_*x1_y1[0]

		y_ = y0+y1+y2+y3
		if y_ == 0:
			y = 0
		else:
			y = y0/y_*x0_y0[1]+y1/y_*x0_y1[1]+y2/y_*x1_y0[1]+y3/y_*x1_y1[1]

		return np.array([x, y])


	def is_perform(self, execution, inexecution):
		return random.choices([True, False], weights=[execution, inexecution])[0]

	def get_margin_scale(self, min_, max_, clip_add_margin, new_shape):
		if clip_add_margin < 0:
			# raise Exception('add margin error')
			return -1, -1
		if min_-clip_add_margin//2 > 0 and max_+clip_add_margin//2 < new_shape:
			if clip_add_margin%2 == 0:
				clip_subtract_margin, clip_plus_margin = clip_add_margin//2, clip_add_margin//2
			else:
				clip_subtract_margin, clip_plus_margin = clip_add_margin//2, clip_add_margin//2+1
		elif min_-clip_add_margin//2 < 0 and max_+clip_add_margin//2 <= new_shape:
			clip_subtract_margin = min_
			clip_plus_margin = clip_add_margin-clip_subtract_margin
		elif max_+clip_add_margin//2 > new_shape and min_-clip_add_margin//2 >= 0:
			clip_plus_margin = new_shape-max_
			clip_subtract_margin = clip_add_margin-clip_plus_margin
		else:
			# raise Exception('add margin error')
			return -1, -1
		return clip_subtract_margin, clip_plus_margin

	# class perturbedCurveImg(object):
	# 	def __init__(self):

	def adjust_position(self, x_min, y_min, x_max, y_max):
		if (self.new_shape[0] - (x_max - x_min)) % 2 == 0:
			f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = (self.new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if (self.new_shape[1] - (y_max - y_min)) % 2 == 0:
			f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = (self.new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0 + 1

		# return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
		return f_g_0_0, f_g_1_0, self.new_shape[0] - f_g_0_1, self.new_shape[1] - f_g_1_1

	def adjust_position_v2(self, x_min, y_min, x_max, y_max, new_shape):
		if (new_shape[0] - (x_max - x_min)) % 2 == 0:
			f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = (new_shape[0] - (x_max - x_min)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if (new_shape[1] - (y_max - y_min)) % 2 == 0:
			f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = (new_shape[1] - (y_max - y_min)) // 2
			f_g_1_1 = f_g_1_0 + 1

		# return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1
		return f_g_0_0, f_g_1_0, new_shape[0] - f_g_0_1, new_shape[1] - f_g_1_1

	def adjust_border(self, x_min, y_min, x_max, y_max, x_min_new, y_min_new, x_max_new, y_max_new):
		if ((x_max - x_min) - (x_max_new - x_min_new)) % 2 == 0:
			f_g_0_0 = ((x_max - x_min) - (x_max_new - x_min_new)) // 2
			f_g_0_1 = f_g_0_0
		else:
			f_g_0_0 = ((x_max - x_min) - (x_max_new - x_min_new)) // 2
			f_g_0_1 = f_g_0_0 + 1

		if ((y_max - y_min) - (y_max_new - y_min_new)) % 2 == 0:
			f_g_1_0 = ((y_max - y_min) - (y_max_new - y_min_new)) // 2
			f_g_1_1 = f_g_1_0
		else:
			f_g_1_0 = ((y_max - y_min) - (y_max_new - y_min_new)) // 2
			f_g_1_1 = f_g_1_0 + 1

		return f_g_0_0, f_g_0_1, f_g_1_0, f_g_1_1

	def interp_weights(self, xyz, uvw):
		tri = qhull.Delaunay(xyz)
		simplex = tri.find_simplex(uvw)
		vertices = np.take(tri.simplices, simplex, axis=0)
		# pixel_triangle = pixel[tri.simplices]
		temp = np.take(tri.transform, simplex, axis=0)
		delta = uvw - temp[:, 2]
		bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
		return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

	def interpolate(self, values, vtx, wts):
		return np.einsum('njk,nj->nk', np.take(values, vtx, axis=0), wts)

	def pad(self, synthesis_perturbed_img_map, x_min, y_min, x_max, y_max):
		synthesis_perturbed_img_map[x_min - 1, y_min:y_max] = synthesis_perturbed_img_map[x_min, y_min:y_max]
		synthesis_perturbed_img_map[x_max + 1, y_min:y_max] = synthesis_perturbed_img_map[x_max, y_min:y_max]
		synthesis_perturbed_img_map[x_min:x_max, y_min - 1] = synthesis_perturbed_img_map[x_min:x_max, y_min - 1]
		synthesis_perturbed_img_map[x_min:x_max, y_max + 1] = synthesis_perturbed_img_map[x_min:x_max, y_max + 1]
		synthesis_perturbed_img_map[x_min - 1, y_min - 1] = synthesis_perturbed_img_map[x_min, y_min]
		synthesis_perturbed_img_map[x_min - 1, y_max + 1] = synthesis_perturbed_img_map[x_min, y_max]
		synthesis_perturbed_img_map[x_max + 1, y_min - 1] = synthesis_perturbed_img_map[x_max, y_min]
		synthesis_perturbed_img_map[x_max + 1, y_max + 1] = synthesis_perturbed_img_map[x_max, y_max]

		return synthesis_perturbed_img_map

	def isSavePerturbed(self, synthesis_perturbed_img, new_shape):
		if np.sum(synthesis_perturbed_img[:, 0]) != 771 * new_shape[0] or np.sum(synthesis_perturbed_img[:, new_shape[1] - 1]) != 771 * new_shape[0] or \
				np.sum(synthesis_perturbed_img[0, :]) != 771 * new_shape[1] or np.sum(synthesis_perturbed_img[new_shape[0] - 1, :]) != 771 * new_shape[1]:
			# raise Exception('clip error')
			return False
		else:
			return True

	def get_angle(self, A, o, B):
		v1 = o-A
		v2 = o-B
		return np.arccos((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))*180/np.pi

	def get_angle_4(self, pts):
		a0_ = self.get_angle(pts[2], pts[0], pts[1])
		a1_ = self.get_angle(pts[0], pts[1], pts[3])
		a2_ = self.get_angle(pts[3], pts[2], pts[0])
		a3_ = self.get_angle(pts[1], pts[3], pts[2])
		return a0_, a1_, a2_, a3_


	def HSV_v1(self, synthesis_perturbed_img_clip_HSV):
		synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_RGB2HSV)
		img_h = synthesis_perturbed_img_clip_HSV[:, :, 0].copy()
		# img_s = synthesis_perturbed_img_clip_HSV[:, :, 1].copy()
		img_v = synthesis_perturbed_img_clip_HSV[:, :, 2].copy()

		if self.is_perform(0.2, 0.8):
			img_h = (img_h + (random.random()-0.5) * 360) % 360  # img_h = np.minimum(np.maximum(img_h+20, 0), 360)
		else:
			img_h = (img_h + (random.random()-0.5) * 40) % 360
		# img_s = np.minimum(np.maximum(img_s-0.2, 0), 1)
		img_v = np.minimum(np.maximum(img_v + (random.random()-0.5)*60, 0), 255)
		# img_v = cv2.equalizeHist(img_v.astype(np.uint8))
		synthesis_perturbed_img_clip_HSV[:, :, 0] = img_h
		# synthesis_perturbed_img_clip_HSV[:, :, 1] = img_s
		synthesis_perturbed_img_clip_HSV[:, :, 2] = img_v

		synthesis_perturbed_img_clip_HSV = cv2.cvtColor(synthesis_perturbed_img_clip_HSV, cv2.COLOR_HSV2RGB)

		return synthesis_perturbed_img_clip_HSV
import numpy as np


def normalize(vector):
	"""Return a normalized copy of a three-dimensional vector."""
	array = np.asarray(vector, dtype=float)
	norm = np.linalg.norm(array)
	if norm < 1e-10:
		raise ValueError("Cannot normalize a zero vector")
	return array / norm


def default_row_direction(direction):
	"""Build a stable in-plane axis perpendicular to an LED direction."""
	direction = normalize(direction)
	row_direction = np.cross(direction, [0.0, 0.0, 1.0])
	if np.linalg.norm(row_direction) < 1e-6:
		row_direction = np.cross(direction, [0.0, 1.0, 0.0])
	return normalize(row_direction)


def rotate_vector(vector, axis, angle_deg):
	"""Rotate a vector around an axis using Rodrigues' rotation formula."""
	vector = np.asarray(vector, dtype=float)
	axis = normalize(axis)
	angle_rad = np.radians(float(angle_deg))
	cosine = np.cos(angle_rad)
	sine = np.sin(angle_rad)
	return (
		vector * cosine
		+ np.cross(axis, vector) * sine
		+ axis * np.dot(axis, vector) * (1.0 - cosine)
	)


def rotation_matrix_x(angle_deg):
	angle_rad = np.radians(float(angle_deg))
	cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
	return np.array(
		[[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
	)


def rotation_matrix_y(angle_deg):
	angle_rad = np.radians(float(angle_deg))
	cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
	return np.array(
		[[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
	)


def rotation_matrix_z(angle_deg):
	angle_rad = np.radians(float(angle_deg))
	cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
	return np.array(
		[[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
	)


def local_frame(forward):
	"""Return orthonormal (forward, horizontal, vertical) axes."""
	forward = normalize(forward)
	reference = np.array([0.0, 0.0, 1.0])
	horizontal = np.cross(forward, reference)
	if np.linalg.norm(horizontal) < 1e-10:
		horizontal = np.cross(forward, [0.0, 1.0, 0.0])
	horizontal = normalize(horizontal)
	vertical = normalize(np.cross(forward, horizontal))
	return forward, horizontal, vertical


def emission_frame(direction):
	"""Return (x_axis, y_axis, z_axis) with z along ``direction``.

	This is the frame used by the ray tracers to map local emission
	directions into world space; kept identical to the historical
	implementation so Monte Carlo results remain reproducible.
	"""
	z_axis = np.asarray(direction, dtype=float)
	if abs(z_axis[2]) < 0.9:
		x_axis = np.cross(z_axis, [0, 0, 1])
	else:
		x_axis = np.cross(z_axis, [0, 1, 0])
	x_axis = x_axis / np.linalg.norm(x_axis)
	y_axis = np.cross(z_axis, x_axis)
	return x_axis, y_axis, z_axis


def as_vec3(v):
	return np.asarray(v, dtype=float).reshape(3)


def rodrigues_rotation(axis, theta_rad):
	"""3x3 rotation matrix about ``axis`` by ``theta_rad`` (Rodrigues)."""
	u = as_vec3(axis)
	n = np.linalg.norm(u)
	if n < 1e-12 or abs(theta_rad) < 1e-15:
		return np.eye(3)
	u = u / n
	c, s = np.cos(theta_rad), np.sin(theta_rad)
	t = 1.0 - c
	ux, uy, uz = u
	return np.array([
		[t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy],
		[t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux],
		[t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c],
	])


def euler_xyz_matrix(roll_deg, pitch_deg, yaw_deg):
	"""Extrinsic X-Y-Z rotation: R = Rz @ Ry @ Rx."""
	return (
		rotation_matrix_z(yaw_deg)
		@ rotation_matrix_y(pitch_deg)
		@ rotation_matrix_x(roll_deg)
	)


def euler_xyz_from_matrix(R):
	"""Extract extrinsic X-Y-Z Euler degrees (roll, pitch, yaw) from R = Rz @ Ry @ Rx."""
	R = np.asarray(R, dtype=float)
	cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
	if cy > 1e-8:
		pitch = np.degrees(np.arctan2(-R[2, 0], cy))
		roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
		yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
	else:
		pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
		roll = np.degrees(np.arctan2(-R[0, 1], R[1, 1]))
		yaw = 0.0
	return float(roll), float(pitch), float(yaw)


def quaternion_to_matrix(wxyz):
	"""3x3 rotation matrix from a (w, x, y, z) quaternion."""
	qw, qx, qy, qz = wxyz
	return np.array([
		[1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
		[2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
		[2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
	])


def quaternion_about_z(angle_deg):
	"""(w, x, y, z) quaternion for a rotation about +Z."""
	half = np.radians(float(angle_deg)) / 2.0
	return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))


def quaternion_multiply(q1, q2):
	"""Hamilton product q1 ⊗ q2 for (w, x, y, z) quaternions."""
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	return (
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
	)


# Reflection across the XZ plane (Y -> -Y).
XZ_MIRROR = np.diag([1.0, -1.0, 1.0])


def mirror_xz_vec(v):
	"""Return (x, -y, z) as a tuple of floats."""
	arr = XZ_MIRROR @ as_vec3(v)
	return (float(arr[0]), float(arr[1]), float(arr[2]))


def mirror_xz_vecs(vecs):
	return [mirror_xz_vec(v) for v in vecs]


def apply_axis_orbit(points, origin, axis, theta_deg):
	"""Orbit 3D points around ``origin``/``axis`` by ``theta_deg``."""
	theta = float(theta_deg)
	if abs(theta) < 1e-12:
		return [tuple(as_vec3(p)) for p in points]
	R = rodrigues_rotation(axis, np.radians(theta))
	O = as_vec3(origin)
	return [tuple(O + R @ (as_vec3(p) - O)) for p in points]


def fit_circle_3points(a, b, c):
	"""Circumcircle of three 3D points -> (center, radius, unit_normal) or None."""
	a, b, c = as_vec3(a), as_vec3(b), as_vec3(c)
	ab = b - a
	ac = c - a
	n = np.cross(ab, ac)
	n_norm = np.linalg.norm(n)
	if n_norm < 1e-10:
		return None
	n = n / n_norm
	mid_ab = (a + b) * 0.5
	mid_ac = (a + c) * 0.5
	d1 = np.cross(n, ab)
	d2 = np.cross(n, ac)
	A = np.column_stack((d1, -d2, n))
	rhs = mid_ac - mid_ab
	try:
		stn, _, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
	except np.linalg.LinAlgError:
		return None
	center = mid_ab + stn[0] * d1
	radius = float(np.linalg.norm(center - a))
	if radius < 1e-6:
		return None
	return center, radius, n


def fit_circle_npoints(points):
	"""Least-squares 3D circle for 3+ points (plane SVD + algebraic 2D fit)."""
	pts = np.asarray(points, dtype=float).reshape(-1, 3)
	if pts.shape[0] < 3:
		return None
	if pts.shape[0] == 3:
		return fit_circle_3points(pts[0], pts[1], pts[2])
	centroid = pts.mean(axis=0)
	centered = pts - centroid
	try:
		_, _, vh = np.linalg.svd(centered, full_matrices=False)
	except np.linalg.LinAlgError:
		return None
	normal = vh[-1]
	n_norm = np.linalg.norm(normal)
	if n_norm < 1e-12:
		return None
	normal = normal / n_norm
	if abs(normal[2]) < 0.9:
		x_axis = np.cross(normal, [0.0, 0.0, 1.0])
	else:
		x_axis = np.cross(normal, [0.0, 1.0, 0.0])
	x_n = np.linalg.norm(x_axis)
	if x_n < 1e-12:
		return None
	x_axis = x_axis / x_n
	y_axis = np.cross(normal, x_axis)
	xy = np.column_stack((centered @ x_axis, centered @ y_axis))
	x, y = xy[:, 0], xy[:, 1]
	A = np.column_stack((x, y, np.ones(len(x))))
	b = -(x * x + y * y)
	try:
		sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
	except np.linalg.LinAlgError:
		return None
	ac, bc, cc = sol
	cx, cy = -0.5 * ac, -0.5 * bc
	r2 = cx * cx + cy * cy - cc
	if r2 <= 1e-12:
		return None
	radius = float(np.sqrt(r2))
	center = centroid + cx * x_axis + cy * y_axis
	return center, radius, normal


def circle_polyline(center, radius, normal, n_seg=64):
	"""(n_seg, 2, 3) line segments approximating a 3D circle (same units as inputs)."""
	center = as_vec3(center)
	nrm = as_vec3(normal)
	n_norm = np.linalg.norm(nrm)
	if n_norm < 1e-12 or radius < 1e-9:
		return np.zeros((0, 2, 3))
	nrm = nrm / n_norm
	if abs(nrm[2]) < 0.9:
		x_axis = np.cross(nrm, [0.0, 0.0, 1.0])
	else:
		x_axis = np.cross(nrm, [0.0, 1.0, 0.0])
	x_axis = x_axis / np.linalg.norm(x_axis)
	y_axis = np.cross(nrm, x_axis)
	thetas = np.linspace(0.0, 2.0 * np.pi, int(n_seg) + 1)
	pts = center + float(radius) * (
		np.cos(thetas)[:, None] * x_axis + np.sin(thetas)[:, None] * y_axis
	)
	return np.stack([pts[:-1], pts[1:]], axis=1)

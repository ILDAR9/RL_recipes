import torch
import matplotlib.pyplot as plt
import math

def estimate_pi():
	n_point = 3000
	points = torch.rand((n_point, 2)) * 2 - 1
	points_circle = []

	for point in points:
		r = torch.sqrt(point[0]**2 + point[1]**2)
		if r <= 1:
			points_circle.append(point)

	points_circle = torch.stack(points_circle)
	plt.plot(points[:, 0].numpy(), points[:, 1].numpy(), 'y.')
	plt.plot(points_circle[:, 0].numpy(), points_circle[:, 1].numpy(), 'c.')

	i = torch.linspace(0, 2 * math.pi, steps = 100)
	plt.plot(torch.cos(i).numpy(), torch.sin(i).numpy())
	plt.axes().set_aspect('equal')
	plt.show()

	pi_estimate = 4 * (len(points_circle) / n_point)
	print("Оценка значения pi:", pi_estimate)

def estimate_pi_mc(n_iteration):
	n_point_circle = 0
	pi_iteration = []
	for i in range(1, n_iteration+1):
		point = torch.rand(2) * 2 - 1
		r = torch.sqrt(point[0] ** 2 + point[1] ** 2)
		if r <= 1:
			n_point_circle += 1
		pi_iteration.append(4 * (n_point_circle / i))
	plt.plot(pi_iteration)
	plt.plot([math.pi] * n_iteration, '--')
	plt.xlabel("Итерация")
	plt.ylabel("Оценка pi")
	plt.title("История оценивания")
	plt.show()
	print("Оценка значения pi:", pi_iteration[-1])


if __name__ == "__main__":
	# estimate_pi()
	estimate_pi_mc(10000)
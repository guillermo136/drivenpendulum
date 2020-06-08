#!/usr/bin/env python3
import driven_pendulum as dp
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def plot_bifurcation(cons):
	plt.figure(figsize=(15, 10))
	min_a = cons.A
	if cons.randInit:
		cons.numIter *= cons.metaCycles
		cons.cycles //= cons.metaCycles
	for i in range(cons.numIter):
		if cons.randInit:
			cons.y0 = np.random.normal(0, 2)
			cons.theta0 = np.random.uniform(-np.pi, np.pi)
		theta, y = dp.simulate_bg(cons)
		offset = int(cons.phi * cons.steps / (2 * np.pi))
		start = cons.delay * cons.steps + offset
		result = np.array(theta)[start::cons.steps]
		plt.plot(cons.A * np.ones_like(result), result, '.', markersize=1, color='red')
		if not cons.randInit or i % cons.metaCycles == 0:
			cons.A += cons.dA
	plt.xlim([min_a - cons.dA, cons.A])
	plt.ylim([-np.pi, np.pi])
	plt.xlabel('Driving Amplitude (A)')
	plt.ylabel('Angle on Poincare Section')
	plt.title('Bifurcation Diagram')
	plt.subplots_adjust(right=0.85)
	place_text(plt, cons)
	if cons.filepath is None:
		plt.show()
	else:
		plt.savefig(cons.filepath)
		plt.close()


def place_text(my_plt, cons=None):
	props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
	textstr = '\n'.join((
		r'$\omega=%.3f$' % cons.w,
		r'$b=%.3f$' % cons.b,
		r'$\phi=%.3f$' % cons.phi))
	my_plt.text(cons.A, 0, textstr, bbox=props)


if __name__ == "__main__":
	parser = ArgumentParser(description="Creates a bifurcation diagram based on the behavior of a driven/damped pendulum. The bifurcation diagram is a plot of the cross sections of the Poincaré map for increasing values of the drove force amplitude.")
	parser.add_argument('--b', default=0.5, type=float, help="Value of damping force coefficient (Default is 0.5)")
	parser.add_argument('--m', default=1, type=float, help="Mass of the pendulum bob (Default is 1)")
	parser.add_argument('--l', default=1, type=float, help="Length of the pendulum rod (Default is 1)")
	parser.add_argument('--g', default=1, type=float, help="Value of acceleration due to gravity (Default is 1)")
	parser.add_argument('--A', default=1, type=float, help="Smallest amplitude of the driving force used in creating the bifurcation diagram (Default is 1)")
	parser.add_argument('--w', default=0.667, type=float, help="Frequency of the driving force (Default is 0.667)")
	parser.add_argument('--randInit', action='store_true', help="If supplied, randomly initiates the initial condition for each simulation. Also runs five simulations per value of A, each of which with one-fifth as many cycles as supplied in cycles.")
	parser.add_argument('--y0', default=1, type=float, help="The initial angular velocity of the pendulum if rabdInit is not used (Default is 1)")
	parser.add_argument('--theta0', default=1.5, type=float, help="The initial angular velocity of the pendulum if rabdInit is not used (Default is 1.5)")
	parser.add_argument('--metaCycles', default=5, type=int, help="The number of different random intializations that will be simulated for each value of A, if using randInit (Default is 5).")
	parser.add_argument('--steps', default=50, type=int, help="Number of timesteps per one cylce of the driving force (Default is 50)")
	parser.add_argument('--delay', default=20, type=int, help="Number of cycles of the driving force to wait before points are plotted on the bifurcation diagram (Default is 20)")
	parser.add_argument('--cycles', default=100, type=int, help="Number of cycles to be run for each value of, excluding the delay (Default is 100)")
	parser.add_argument('--phi', default=0, type=float, help="Phase shift of the Poincaré map to be used for plotting the bifurcation diagram (Default is 0)")
	parser.add_argument('--dA', default=0.005, type=float, help="Difference between each value of A that is simulated (Default is 0.005)")
	parser.add_argument('--numIter', default=400, type=int, help="Number of different values of A that are simulated to create the bifurcation diagram (Default is 400)")
	parser.add_argument('--filepath', default=None, type=str, help="If supplied, filepath saves the bifurcation diagram to this filepath.")

	args = parser.parse_args()
	dp.set_timesteps(args)
	args.noShift = False

	plot_bifurcation(args)
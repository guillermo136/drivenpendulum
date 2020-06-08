#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from argparse import ArgumentParser

plt.style.use("ggplot")


def configure_pendulum_ax(ax):
    ax.set_theta_zero_location("S")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_ylim([0, 1.2])
    rod, = ax.plot([], [], color='b')
    weight, = ax.plot([], [], color='b', marker='o', markersize=24)
    arrow_shaft, = ax.plot([], [], '--', color='r', alpha=1, linewidth=1)
    arrow_head, = ax.plot([], [], '--', color='r', linewidth=1)
    return rod, weight, arrow_shaft, arrow_head


def configure_graph_ax(ax, title, cons=None, marker='.', m_size=1, put_text=False):
    graph, = ax.plot([], [], marker, markersize=m_size)
    ax.set_ylim([-5, 5])
    ax.set_xlabel("theta")
    ax.set_ylabel("y")
    ax.set_title(title)
    if put_text:
        place_text(ax, cons)
    return graph


def place_text(ax, cons=None):
    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    textstr = '\n'.join((
        r'$A=%.3f$' % (cons.A / cons.g),
        r'$\omega=%.3f$' % cons.w,
        r'$b=%.3f$' % cons.b))
    ax.text(1.05, 0.5, textstr, transform=ax.transAxes, bbox=props)


def configure_slider(plt, ax):
    pos = ax.get_position()
    ax = plt.axes([pos.x0, pos.y0 - 0.13, pos.x1 - pos.x0, 0.02])
    s_phi = Slider(ax, 'phi', 0, 2 * np.pi, valinit=0, color='blue')
    return s_phi


def shift(theta, cons):
    if cons.noShift:
        return theta
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def reset_axes(ax, x, y, buffer=0.1):
    ax.set_xlim([np.min(x) - buffer, np.max(x) + buffer])
    ax.set_ylim([np.min(y) - buffer, np.max(y) + buffer])


def rk_iter(theta, y, z, cons):
    i = -1
    args = (theta[-1], y[-1], z[-1], cons)
    funcs = [dtheta_dt, dy_dt, dz_dt]
    k1 = []
    for f in funcs:
        k1.append(cons.dt * f(*args))

    k2 = []
    temp_args = [0, 0, 0, cons]
    for i in range(3):
        temp_args[i] = (args[i] + 0.5 * k1[i])
    for f in funcs:
        k2.append(cons.dt * f(*temp_args))

    k3 = []
    for i in range(3):
        temp_args[i] = (args[i] + 0.5 * k2[i])
    for f in funcs:
        k3.append(cons.dt * f(*temp_args))

    k4 = []
    for i in range(3):
        temp_args[i] = (args[i] + k3[i])
    for f in funcs:
        k4.append(cons.dt * f(*temp_args))

    theta.append(shift(theta[-1] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6, cons))
    y.append(y[-1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6)
    z.append(z[-1] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6)


def dtheta_dt(theta, y, z, cons):
    return y


def dy_dt(theta, y, z, cons):
    term1 = (-cons.b / cons.m / cons.l ** 2) * y
    term2 = -cons.g / cons.l * np.sin(theta)
    term3 = cons.A / cons.m / cons.l ** 2 * np.sin(z)
    return term1 + term2 + term3


def dz_dt(theta, y, z, cons):
    return cons.w


def simulate(cons):
    theta = [cons.theta0]
    y = [cons.y0]
    z = [0]

    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(bottom=0.15)
    ax1 = fig.add_subplot(131, polar=True)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    rod, weight, arrow_s, arrow_h = configure_pendulum_ax(ax1)
    phase = configure_graph_ax(ax2, "Phase Diagram", marker='.', m_size=1)
    poincare = configure_graph_ax(ax3, "Poincaré Section", cons=cons, marker='+', m_size=3, put_text=True)
    slider = configure_slider(plt, ax3)

    def animate(i, theta, y, z, cons):
        rk_iter(theta, y, z, cons)

        rod_theta = np.array([theta[-1], theta[-1]])
        rod_r = np.array([0, 1])
        arrow_r = np.ones(20) * 0.8
        arrow_t = np.linspace(0, np.sin(z[-1]), 20)
        head_t = [arrow_t[-1] - 0.1 * np.sign(arrow_t[-1]), arrow_t[-1], arrow_t[-1] - 0.1 * np.sign(arrow_t[-1])]
        head_r = [arrow_r[-1] - 0.05, arrow_r[-1], arrow_r[-1] + 0.05]
        phi = int(slider.val * cons.steps / (2 * np.pi))
        start = cons.delay * cons.steps + phi

        rod.set_data(rod_theta, rod_r)
        weight.set_data(theta[-1], 1)
        phase.set_data(theta, y)
        arrow_s.set_data(arrow_t, arrow_r)
        arrow_h.set_data(head_t, head_r)
        poincare.set_data(np.array(theta)[start::cons.steps], np.array(y)[start::cons.steps])

        reset_axes(ax2, theta, y)
        reset_axes(ax3, theta, y)

    ani = FuncAnimation(fig, animate, fargs=(theta, y, z, cons), interval=1, frames=10000)
    plt.show()


def simulate_bg(cons):
    theta = [cons.theta0]
    y = [cons.y0]
    z = [0]
    for i in range(cons.steps * (cons.delay + cons.cycles)):
        rk_iter(theta, y, z, cons)
    return theta, y

def static_plot(cons):
    theta, y = simulate_bg(cons)
    if cons.smallPlot:
        size = (8, 4)
    else:
        size = (15, 7)
    fig = plt.figure(figsize=size)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    phase = configure_graph_ax(ax1, "Phase Diagram", marker='.', m_size=1)
    poincare = configure_graph_ax(ax2, "Poincaré Section", cons=cons, marker='+', m_size=3, put_text=True)
    
    phi = int(cons.phi * cons.steps / (2 * np.pi))
    start = cons.delay * cons.steps + phi

    phase.set_data(theta, y)
    poincare.set_data(np.array(theta)[start::cons.steps], np.array(y)[start::cons.steps])

    reset_axes(ax1, theta, y)
    reset_axes(ax2, theta, y)

    if cons.filepath is None:
        plt.show()
    else:
        plt.savefig(cons.filepath)
        plt.close()

def set_timesteps(cons):
    cons.dt = (2 * np.pi / cons.w) / cons.steps

if __name__ == "__main__":
    parser = ArgumentParser(description="Simulate a Driven/Damped Pendulum. Try running without any parameters to see the default behavior.")
    parser.add_argument('--b', default=0.5, type=float, help="Value of damping force coefficient (Default is 0.5)")
    parser.add_argument('--m', default=1, type=float, help="Mass of the pendulum bob (Default is 1)")
    parser.add_argument('--l', default=1, type=float, help="Length of the pendulum rod (Default is 1)") 
    parser.add_argument('--g', default=1, type=float, help="Value of acceleration due to gravity (Default is 1)")
    parser.add_argument('--A', default=1.45, type=float, help="Amplitude of the driving force (Default is 1)")
    parser.add_argument('--w', default=0.667, type=float, help="Frequency of the driving force (Default is 0.667)")
    parser.add_argument('--y0', default=1, type=float, help="Initial angular velocity (Default is 1)")
    parser.add_argument('--theta0', default=1.5, type=float, help="Initial angle of pendulum (Default is 1.5)")
    parser.add_argument('--steps', default=50, type=int, help="Number of timesteps per one cylce of the driving force (Default is 50)")
    parser.add_argument('--delay', default=5, type=int, help="Number of cycles of the driving force before points are plotted on the Poincaré map (Default is 5)")
    parser.add_argument('--noShift', action='store_true', help="noShift allows the angle plotted to not be limited to the range (-pi, pi). Instead the number of net revolutions made by the pendulum is included in the angle.")
    parser.add_argument('--static', action='store_true', help="static simulates the motion of the pendulum without animation, instead only displaying the final result.")
    parser.add_argument('--smallPlot', action='store_true', help="Sets the size of the image displayed by static at 800x400 instead of 1500x700. Ignored if not used with static.")
    parser.add_argument('--cycles', default=100, type=int, help="Number of cycles of the driving force to be simulated before displaying the static image, excluding the delay. Ignored if not used with static. (Default is 100)")
    parser.add_argument('--phi', default=0, type=float, help="The phase shift to be used on the Poincaré map for the static image. Ignored if not used with static. (Default is 0)")
    parser.add_argument('--filepath', default=None, type=str, help="If supplied, filepath saves the static image to this filepath. Ignored if not used with static.")

    args = parser.parse_args()
    set_timesteps(args)
    
    if args.static:
        static_plot(args)
    else :
        simulate(args)

# %%
import numpy as np
import sympy as sp
from scipy.linalg import block_diag

R1a = 0.3
R2a = 0.5
dwa = 0.0
R1b = 70.0
R2b = 30.0
kba = 200.0
fb = 5e-2
dwb = 3.5
offset = 2.0
power = 1.5
B0 = 9.4
gamma = 267.522
tp = 15


def gen_spectrum(R1a, R2a, dwa, R1b, R2b, kba, fb, dwb, offset, power, B0, gamma, tp):
    # R1a, R2a, dwa, R1b, R2b, kba, fb, dwb, offset, power, B0, gamma, tp = sp.symbols(
    #     "R1a, R2a, dwa, R1b, R2b, k, f, dwb, offset, power, B0, gamma, tp", real=True
    # )
    kab = fb * kba
    Pa = kba / (kba + kab)
    Pb = kab / (kba + kab)
    theta_a = np.atan(power / (B0 * (offset - dwa)))
    theta_b = np.atan(power / (B0 * (offset - dwb)))
    omega_a = np.sqrt((gamma * power) ** 2 + (B0 * gamma * (offset - dwa)) ** 2)
    omega_b = np.sqrt((gamma * power) ** 2 + (B0 * gamma * (offset - dwb)) ** 2)

    R2p_a = 0.5 * (R2a * (1 + np.cos(theta_a) ** 2) + R1a * np.sin(theta_a) ** 2)
    R1p_a = R2a * np.sin(theta_a) ** 2 + R1a * np.cos(theta_a) ** 2
    R2p_b = 0.5 * (R2b * (1 + np.cos(theta_b) ** 2) + R1b * np.sin(theta_b) ** 2)
    R1p_b = R2b * np.sin(theta_b) ** 2 + R1b * np.cos(theta_b) ** 2

    def Lx(R2x, R1x, dwx, offset, power, b0, gamma):
        return np.array(
            [
                [-R2x, -(offset - dwx) * b0 * gamma, 0],
                [(offset - dwx) * b0 * gamma, -R2x, -power * gamma],
                [0, power * gamma, -R1x],
            ]
        )

    def L0x(R2p, omega, R1p):
        return np.array([[-R2p - omega * 1j, 0, 0], [0, -R2p - omega * 1j, 0], [0, 0, -R1p]])

    def Dx(theta, R1, R2):
        dR = R2 - R1
        return (
            dR
            / 2
            * np.array(
                [
                    [0, np.sin(theta) ** 2, -1 / np.sqrt(2) * np.sin(2 * theta)],
                    [np.sin(theta) ** 2, 0, -1 / np.sqrt(2) * np.sin(2 * theta)],
                    [-1 / np.sqrt(2) * np.sin(2 * theta), -1 / np.sqrt(2) * np.sin(2 * theta), 0],
                ]
            )
        )

    def Ry(theta):
        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def Ry_bar(theta_a, theta_b):
        dtheta = theta_b - theta_a
        return np.array(
            [
                [0.5 * (np.cos(dtheta) + 1), 0.5 * (np.cos(dtheta) - 1), 1 / np.sqrt(2) * np.sin(dtheta)],
                [0.5 * (np.cos(dtheta) - 1), 0.5 * (np.cos(dtheta) + 1), 1 / np.sqrt(2) * np.sin(dtheta)],
                [-1 / np.sqrt(2) * np.sin(dtheta), -1 / np.sqrt(2) * np.sin(dtheta), np.cos(dtheta)],
            ]
        )

    K = np.kron(np.array([[-kba * fb, kba], [kba * fb, -kba]]), np.eye(3))
    S = np.kron(np.array([[sp.sqrt(Pa), 0], [0, np.sqrt(Pb)]]), np.eye(3))
    R = block_diag(Ry(theta_a), Ry(theta_b))
    u = 1 / np.sqrt(2) * np.array([[1, 1, 0], [1j, -1j, 0], [0, 0, np.sqrt(2)]])
    U = block_diag(u, u)

    Lambda = block_diag(
        L0x(R2p_a, omega_a, R1p_a) - kab * np.eye(3), L0x(R2p_b, omega_b, R1p_b) - kba * np.eye(3)
    )
    Gamma = np.block(
        [
            [Dx(theta_a, R1a, R2a), np.sqrt(kab * kba) * Ry_bar(theta_a, theta_b)],
            [np.sqrt(kab * kba) * Ry_bar(theta_b, theta_a), Dx(theta_b, R1b, R2b)],
        ]
    )
    A = (
        block_diag(
            Lx(R2a, R1a, dwa, offset, power, B0, gamma), Lx(R2b, R1b, dwb, offset, power, B0, gamma)
        )
        + K
    )
    M_ss = -np.linalg.solve(A, np.array([0, 0, R1a, 0, 0, R1b * fb]))
    C = np.array(
        [
            [0.5 * (np.cos(theta_b - theta_a) + 1), 0, 0],
            [0, 0.5 * (np.cos(theta_b - theta_a) + 1), 0],
            [0, 0, np.cos(theta_b - theta_a)],
        ]
    )

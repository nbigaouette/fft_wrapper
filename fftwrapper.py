#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Nicolas Bigaouette
# Fall 2012

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

def Closest_Power_of_Two_Down(N):
    return int(2**(np.floor(np.log(N) / np.log(2.0))))

def Closest_Power_of_Two_Up(N):
    return int(2**(np.ceil(np.log(N) / np.log(2.0))))

class FFT:
    def __init__(self):
        self.resize_NFFT    = 0;
        self.nt             = 0;
        self.nf             = 0;
        self.t              = 0.0;
        self.signal         = 0.0;
        self.dt             = 0.0;
        self.Fs             = 0.0;
        self.f              = 0.0;
        self.omega          = 0.0;
        self.S              = 0.0; # FT(signal)
        self.Sabs           = 0.0;
        self.Sabs2          = 0.0;
        self.phase          = 0.0;

    def Set_Time_Signal(self, t, signal, resize_NFFT = 0):
        self.t              = t
        self.signal         = signal
        self.nt             = len(self.t)
        self.resize_NFFT    = int(resize_NFFT)

        self.dt = self.t[1] - self.t[0]
        self.Fs = 1.0 / self.dt

        if (self.resize_NFFT != 0):
            # Makes FFT algorithm faster by resizing signal

            print("Changing NFFT from", self.nt, "to ", end="")

            if (self.resize_NFFT == -1):
                # -1 will resize down to closest power of two
                self.nt     = Closest_Power_of_Two_Down(self.nt)
                self.signal = self.signal[0:self.nt]
                self.t      = self.t[0:self.nt]
            elif (self.resize_NFFT < self.nt and self.resize_NFFT != 1):
                # Resize down to specific value
                self.nt     = self.resize_NFFT
                self.signal = self.signal[0:self.nt]
                self.t      = self.t[0:self.nt]
            else:
                # Resize up by padding with 0
                old_nt = self.nt
                if (self.resize_NFFT == +1):
                    # +1 will resize up to closest power of two
                    self.nt = Closest_Power_of_Two_Up(self.nt)
                else:
                    # More than one will resize to that exact value
                    assert(self.resize_NFFT > self.nt)
                    self.nt = int(self.resize_NFFT)

                new_signal          = np.zeros(self.nt, dtype=np.complex128)
                new_signal[0:old_nt]= self.signal
                self.signal         = new_signal
                self.t              = np.linspace(self.t[0], (self.nt-1)*self.dt, self.nt)

            print(self.nt)

        assert(len(self.t)      == self.nt)
        assert(len(self.signal) == self.nt)

        # Frequency vector
        N = self.nt
        n = float(N)
        # Equivalent to:
        # fft_freq = self.Fs*np.fft.fftshift(np.fft.fftfreq(self.t.shape[-1]))
        # See http://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html
        if (self.nt % 2 == 0):
            #self.f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)         # if n is even
            self.f = np.concatenate((
                        np.linspace(-n/2.0, -1.0,    n/2),
                        np.linspace( 0.0,   n/2.0-1, n/2)
                    )) * self.Fs / n
        else:
            #self.f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   # if n is odd
            self.f = np.concatenate((
                        np.linspace(-(n-1.0)/2.0, -1.0,           N/2),
                        np.linspace( 0.0,         float(N-1)/2.0, N/2+1)
                    )) * self.Fs / n

        assert(abs((self.f[2] - self.f[1]) - (self.Fs / float(self.nt))) < 1.0e-14)
        self.omega  = 2.0 * np.pi * self.f
        self.nf     = len(self.f)

        # S = TF(s)
        self.S      = np.fft.fftshift(np.fft.fft(self.signal, self.nt))
        self.Derive_From_S()

        self.df     = self.f[1] - self.f[0]
        self.Ts     = 1.0 / self.df

    def Set_Freq_FT(self, frequencies, FT, resize_NFFT = 0):
        self.f              = frequencies
        self.omega          = frequencies * 2.0 * np.pi
        self.S              = FT
        self.nf             = len(self.f)
        self.resize_NFFT    = int(resize_NFFT)

        if (self.nf > 1):
            self.df = self.f[1] - self.f[0]
            self.Ts = 1.0 / self.df

        self.Derive_From_S()

        assert(len(self.f) == self.nf)
        assert(len(self.S) == self.nf)

        # Time vector
        if (self.nf > 1):
            #self.t      = np.linspace(-self.Ts/2.0, self.Ts/2.0, self.nf)
            self.t      = np.linspace(0.0, self.Ts, self.nf)
            self.nt     = len(self.t)

        # s = TFi(S)
        self.signal = np.fft.ifft(np.fft.ifftshift(self.S))

        if (self.nf > 1):
            self.dt = self.t[1] - self.t[0]
            self.Fs = 1.0 / self.dt

    def Derive_From_S(self):
        self.Sabs   = abs(self.S)
        self.Sabs2  = self.Sabs**2
        self.phase  = np.angle(self.S)

    def Plot(self):
        import matplotlib.pyplot as plt

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(223)
        self.ax3 = self.fig.add_subplot(224, sharex=self.ax2)

        self.ax1.plot(self.t, self.signal.real, lw=2, label='Real')
        self.ax1.plot(self.t, self.signal.imag, lw=2, label='Imaginary')
        self.ax1.set_xlabel("t [time]")
        self.ax1.set_ylabel("Signal [arb. unit]")
        leg = self.ax1.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.75)

        # Plot |FFT|
        self.ax2.plot(self.f, self.Sabs, '-', lw=2)
        self.ax2.set_xlabel(r"frequencies [time$^{-1}$]")
        self.ax2.set_ylabel(r"Spectrum")
        self.ax2.set_yscale('log')

        # Plot /_ FFT
        self.ax3.plot(self.f, self.phase, '-', lw=2)
        self.ax3.set_xlabel(r"frequencies [time$^{-1}$]")
        self.ax3.set_ylabel(r"Phase")

        # plt.tight_layout()

        plt.show()

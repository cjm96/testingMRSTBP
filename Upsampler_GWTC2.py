import numpy as np
import matplotlib.pyplot as plt

import os
import json

from scipy.interpolate import interp1d

from pesummary.io import read
from pesummary.gw.fetch import fetch_open_data
from pesummary.gw.file.strain import StrainData

from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries


# path to where public pe data is stored
samples_root = os.path.abspath('all_posterior_samples')

# path to where public strain data is stored
strain_root = os.path.abspath('strain_data')


def my_whiten(strain_timeseries, psd):
    """
    Manually whiten a time series
    """
    fS = np.array(psd)
    S = interp1d(fS[:,0], fS[:,1], bounds_error=False, fill_value=np.inf)

    times = np.array(strain_timeseries.times)
    freqs = np.fft.rfftfreq(len(times), d=np.mean(np.diff(times)))
    strain_td = np.array(strain_timeseries)
    strain_fd = np.fft.rfft(strain_td)
    strain_fd /= np.sqrt(S(freqs))
    strain_td_w = np.fft.irfft(strain_fd)
    return TimeSeries(strain_td_w, times=times)


class Upsampler:
    """
    Class for peforming posterior upsampling on GW events from GWTC-2.

    This uses the statistical method described in [1].

    All GW data comes from [2] which must be downloaded manually
    and stored in the same directory as this file.

    This class makes extensive use of the pesummary package [3].

    [1] Alvin J. K. Chua, "Sampling from manifold-restricted distributions
    using tangent bundle projections", https://arxiv.org/abs/1811.05494,
    Statistics and Computing vol. 30, p.587–602 (2020)
    https://doi.org/10.1007/s11222-019-09907-8

    [2] LIGO Document P2000223-v5, https://dcc.ligo.org/LIGO-P2000223/public

    [3] Charlie Hoy and Vivien Raymond, "PESummary: the code agnostic
    Parameter Estimation Summary page builder", https://arxiv.org/abs/2006.06639
    https://lscsoft.docs.ligo.org/pesummary/igwn_pinned/what_is_pesummary.html
    """

    def __init__(self, event, approx=None, comoving=False):
        """
        INPUTS
        ------
        event: str
            GW event name ID, e.g. 'GW190412'
        approx: str
            name of the waveform approximant to use
        comoving: bool
            if true, then load the _comoving samples
        """
        self.event = event
        self.approx = approx
        self.comoving = comoving

        self.samples_file = os.path.join(samples_root, event+'.h5')

        self.pesummary_data = read(self.samples_file) # pesummary read (~50s)

        self.load_meta_data() # various event meta data (<1s)

        self.load_noise_psd() # noise information (<1s)

        self.load_strain_data() # strain data (~100s unless already in cache)

        self.load_samples() # posterior samples (~100s)

        # derivatives: Jac and Hess

        # upsampling


    def load_meta_data(self):
        """
        Load various meta data for the event
        """
        # approximant
        samples_avail = list(self.pesummary_data.config.keys())

        if self.approx is None:
            Pv2, IMRD = 'C01:IMRPhenomPv2', 'C01:IMRPhenomD'
            self.approx = Pv2 if Pv2 in samples_avail else IMRD
            assert self.approx in samples_avail, ("Available approximants: "\
                                                    + samples_avail)
            print("Available samples = ", samples_avail,
                        "... using", self.approx)
        else:
            test = self.approx in samples_avail
            assert test, self.approx+" not in "+str(samples_avail)

        # interferometers
        ifo_str = self.pesummary_data.config[self.approx]['analysis']['ifos']

        self.IFOs, possible_IFOs = [], ['H1', 'L1', 'V1']
        for x in possible_IFOs:
            if x in ifo_str:
                self.IFOs += [x]

        print("Interferometers used:", self.IFOs)

        # analysis times
        input = self.pesummary_data.config[self.approx]['input']

        self.t_start = int(input['gps-start-time'])
        self.t_end = int(input['gps-end-time'])

        print("Analysis time range: ", (self.t_start, self.t_end))

        # frequencies
        f_ref = self.pesummary_data.config[self.approx]['engine']['fref']
        self.f_ref = float(f_ref)

        f_low = self.pesummary_data.config[self.approx]['lalinference']['flow']
        self.f_low = json.loads(f_low.replace("'", '"'))

        self.delta_f = {IFO: 1./256. for IFO in self.IFOs} # FIXME


    def load_noise_psd(self):
        """
        Load the noise PSDs for the event
        """
        self.psd = self.pesummary_data.psd[self.approx]

        self.f_high = {}
        for IFO in self.IFOs:
            self.f_high[IFO] = np.array(self.psd[IFO])[-1,0]


    def load_strain_data(self, cache=True):
        """
        Load the strain data for the event

        If it doesn't exist already, then it will be downloaded
        into the strain_data folder

        INPUTS
        ------
        cache: bool
            if True, then store local copy of strain data
            passed to pesummary StrainData.fetch_open_data
        """
        self.data = {}
        self.data_w = {}
        for IFO in self.IFOs:
            # download public strain data
            print("Downloading strain data for "+IFO)
            self.data[IFO] = StrainData.fetch_open_data(IFO,
                                                self.t_start, self.t_end,
                                                cache=cache)

            # unpack PSD into FrequencySeries ASD object
            fS = np.array(self.psd[IFO])
            asd = FrequencySeries(np.sqrt(fS[:,1]), frequencies=fS[:,0])

            # whitened strain
            self.data_w[IFO] = self.data[IFO].whiten(asd=asd)


    def load_samples(self):
        """
        Load the posterior samples for the event
        """
        print("Loading pesummary samples dictionary")
        self.samples = self.pesummary_data.samples_dict[self.approx]

        # store number of samples and index of maximum posterior sample
        self.MAPindex = np.argmax(self.samples['logpost'])
        self.Nsamples = len(self.samples[list(self.samples.keys())[0]])


    def waveform(self, IFO=None, index=None, whiten=True):
        """
        Generate waveform from a posterior sample

        INPUTS
        ------
        IFO: str
            which instrument into which to project
            e.g. 'H1', 'L1' or 'V1'
        index: int
            which individual posterior sample to use?
            if None, then use the max posterior point
        whiten: bool
            if True, then whiten the signal

        RETURNS
        -------
        model:
            the model waveform
        """
        ind = self.MAPindex if index is None else ind

        dt = 1/2**np.ceil(np.log2(2*max(self.f_high.values())))

        # compute TD waveform
        model = self.samples.td_waveform(self.approx, dt,
                    self.f_low[IFO], f_ref=self.f_ref,
                    ind=ind, project=IFO)
        model.dx = dt

        if whiten:
            # unpack PSD into FrequencySeries ASD object
            fS = np.array(self.psd[IFO])
            asd = FrequencySeries(np.sqrt(fS[:,1]), frequencies=fS[:,0])

            # whiten the model
            model = model.whiten(asd=asd)

        return model


    def plot_MAP_waveform(self, outname='MAPwaveform.png',
                                bandpass=(30, 200),
                                tlim=(-1, 1), ylim=(-5,5)):
        """
        Plot the maximum aposteriori waveform on top of the
        whitened strain data for each instrument.

        Also compute the log-likelihood values for this waveform.

        INPUTS
        ------
        outname: str
            output file name for plot
        bandpass: tuple
            bandpass filtering range
        tlim: tuple
            time range to plot relative to peak time in L1
        ylim: tuple
            whitened strain range to plot
        """
        # reference time
        t0 = np.mean(self.samples['L1_time'])

        # check output directory exists
        outdir = os.path.join('output', self.event)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        # output file name
        outfile = os.path.join(outdir, outname)
        print("Plotting MAP waveform: saving to "+outfile)

        # create the figure
        fig, axs = plt.subplots(len(self.IFOs))

        # loop over the instruments...
        for i, IFO in enumerate(self.IFOs):

            # unpack PSD into FrequencySeries ASD object
            fS = np.array(self.psd[IFO])
            asd = FrequencySeries(np.sqrt(fS[:,1]), frequencies=fS[:,0])

            # ... plotting the data ...
            d = self.data[IFO]
            axs[i].plot(np.array(d.times) - t0,
                        np.array(d.bandpass(*bandpass).whiten(asd=asd)),
                        c='k', ls='-', lw=1, alpha=0.5)

            # ... and the model.
            m = self.waveform(IFO=IFO, whiten=False)
            axs[i].plot(np.array(m.times) - t0,
                        np.array(m.bandpass(*bandpass).whiten(asd=asd)),
                        c='cyan', ls='-', lw=1, alpha=1,
                        label=IFO)

            # plot decorations
            axs[i].legend(loc='upper left', frameon=False)

            axs[i].set_xlim(*tlim)
            #axs[i].set_ylim(*ylim)

            if i==1:
                axs[i].set_ylabel("Whitened Strain")

            if i == len(self.IFOs)-1:
                axs[i].set_xlabel("Time from {} [s]".format(t0))
            else:
                axs[i].set_xticklabels([])

        # put title on figure
        fig.suptitle(self.event)# + ": max log-like = {}".format())

        # save output
        plt.savefig(outfile)
        plt.clf()


    def log_like(self, Ntest):
        """
        Pick some posterior samples and compute the likelihood,
        compare with the value given in the public samples.

        INPUTS
        ------
        Ntest: int
            how many random test samples to use

        RETURNS
        -------
        log_likes: array
            the calculated and samples likelihoods
        """

        test_samples = np.random.choice(np.arange(self.Nsamples),
                                        Ntest, replace=False)

        log_likes = np.zeros((Ntest, 2))

        for i, ind in enumerate(test_samples):
            loglike=0
            for IFO in self.IFOs:
                model = waveform(self, IFO=IFO, index=ind)
                

                loglike += -0.5*np.abs(d - np.array(model) )**2

            log_likes[i] = [loglike,
                                self.samples['log_likelihood'][ind]]

        return log_likes



if __name__ == "__main__":

    event = 'GW190521'
    GW = Upsampler(event, approx='C01:IMRPhenomPv2')
    GW.plot_MAP_waveform(bandpass=(20, 250), tlim=(-0.25, 0.05))

    event = 'GW190412'
    GW = Upsampler(event, approx='C01:IMRPhenomPv3HM')

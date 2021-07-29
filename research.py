"""
Nicholas Summerfield's research code.
EventOutput is used to analyze hic-eventgen outputs
TrentoOutput is for Trento
HydroOutput is for the hydro evolution

Updated: 04/11/2021
"""
import os
import numpy as np
import math as m
from scipy import stats
import pandas as pd
import warnings as w
from PIL import Image
import sys


def dir_check(output_directory, png=False, pdf=False):
    # Checking if a file exists. If it doesn't, this will create the file.
    if png == True and pdf == True:
        print('Can only save one type at a time. Please select EITHER png or pdf')
        return
    print('Saving data to: {}'.format(output_directory))
    path = ''
    elements = output_directory[1:].split('/')
    if png == True:
        elements.append('png')
    if pdf == True:
        elements.append('pdf')
    for i, level in enumerate(elements):
        path = '/'.join([path, elements[i]])
        if os.path.isdir('{}'.format(path)) == False:
            os.mkdir('{}'.format(path))
    return


def toGIF(OutPut, name, FileCount = 0, Duration = 200):
    path = '{}'.format(OutPut) + '/png'
    frames = []
    if FileCount == 0:
        for file in os.listdir(path):
            if not file.startswith('.'):
                FileCount += 1

    for i in range(FileCount):
        frames.append(Image.open(os.path.join(path, 'fig{0:0=4d}.png'.format(i))))
    frames[0].save('{}'.format(OutPut) + '/{}.gif'.format(name), format='GIF', append_images=frames[1:], save_all=True, duration=Duration, loop=0)
    return


# My x axis:
def X_():
    # This returns the appropriate array for the X_ values used in most of the plots.
    return [90, 75, 65, 55, 45, 35, 25, 15, 7.5, 2.5]

class EventOutput:
    # Built for HIC-eventgen output, see https://github.com/Duke-QCD/hic-eventgen for more
    def __init__(self, RawFiles):
        # Pulling the data from the files. This is expecting a list of files. If you are only looking at one file, put it in a bracket.
        # These are the labels that organize the data from the HIC-eventgen output
        species = [('pion', 211),('kaon', 321),('proton', 2212),('Lambda', 3122),('Sigma0', 3212),('Xi', 3312),('Omega', 3334)]
        EventDtype = [('initial_entropy', '<f8'),('nsamples', '<i8'),('dNch_deta', '<f8'),('dET_deta', '<f8'),('dN_dy', [(s, '<f8') for (s, _) in species]),('mean_pT', [(s, '<f8') for (s, _) in species]),('pT_fluct', [('N', '<i8'), ('sum_pT', '<f8'), ('sum_pTsq', '<f8')]),('flow', [('N', '<i8'), ('Qn', '<c16', 8)])]
        events = np.empty([0], dtype=EventDtype)
        for path in RawFiles:
            for file in os.listdir(path):
                if not file.startswith('.'):
                    path_file = os.path.join(path, file)
                    events_file = np.fromfile(path_file, dtype=EventDtype)
                    events = np.concatenate((events, events_file), axis=0)
        self.FullAray = events

    def Centrality(self):
        InitialEntropy = self.FullAray['initial_entropy'] # Getting the Entropy information from the

        MaxEntropy = m.ceil(max(InitialEntropy)) # Defining the maximum entropy value for this data set
        CentralityMatrix = np.ones([2,9]) #, dtype=[('Entropy', float),('Percent', float)])
        CentralityCalculation = np.ones([3,MaxEntropy]) #, dtype=[('initial_entropy_histogram', float),('initial_entropy_sum', float),('binned_entropy', float)])
        BinPercentages = [0, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1]
        PercentageBins = np.diff(BinPercentages) / 2 + np.array(BinPercentages[:(len(BinPercentages) - 1)])
        CentralityMatrix[1] = 100 * (1 - np.array(BinPercentages[1:10])) # Putting in the Centrality Percent Values to the final matrix

        CentralityCalculation[0], bins = np.histogram(InitialEntropy, bins=MaxEntropy, range=(0, MaxEntropy), density=True)
        # Creating a Histogram that counts all the instances for each individual entropy integer and normalizes it to the total number of events
        for i in range(MaxEntropy):
            CentralityCalculation[1][i] = sum(CentralityCalculation[0][:i + 1])
        # Sums up the previous histogram turning each position into a percentage of total entropy
        CentralityCalculation[2] = pd.cut(CentralityCalculation[1], bins=BinPercentages, labels=PercentageBins)
        # Groups the percentages into the actual bins

        WorkingArray = []
        for i in range(9): # For each bin
            for ii in range(len(CentralityCalculation[0])): # For each individual entropy instance
                if CentralityCalculation[2][ii] == PercentageBins[i]: # If it falls into the desired bin, add it to the list
                    WorkingArray = np.append(WorkingArray, ii + 1)
            CentralityMatrix[0][i] = max(WorkingArray) # Take the cut off entropy for that given bin
        return np.flip(CentralityMatrix, 1) # Reverses the order so the data starts at low centrality

    def pT(self, bins=None):
        """
        Taking the events defined in EventOutput() and calculating the <pT>
        :returns 10x8 matrix. [:,0] = pion, [:,1] = kaon, [:,2] = proton, [:,3] = charged, rest = errors.
        Optional, externally define the centrality bins.
        """
        # Defining the bins to compare the data to. If left blank, uses the bins determined by the data set
        if bins is None:
            bins = np.flip(self.Centrality()[0])

        w.filterwarnings("ignore")
        Event = self.FullAray
        CentralityBins = np.append(0, np.append(bins ,m.ceil(max(Event['initial_entropy']))))

        pT = np.ones([10, 8])
        # 10 bins, one for the four types and one for each of their errors. This is the final output matrix
        # Pion, kaon, proton, charged particle, pion error, kaon error, proton error, charged error
        # 0     1     2       3                 4           5           6             7
        pT[:, 0], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['mean_pT']['pion'], bins=CentralityBins)
        pT[:, 1], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['mean_pT']['kaon'], bins=CentralityBins)
        pT[:, 2], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['mean_pT']['proton'], bins=CentralityBins)

        # Binning the multiplicities. This is used for the charged <pT> Calculation
        dNdy = np.ones([10, 3])
        dNdy[:, 0], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['dN_dy']['pion'], bins=CentralityBins)
        dNdy[:, 1], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['dN_dy']['kaon'], bins=CentralityBins)
        dNdy[:, 2], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['dN_dy']['proton'], bins=CentralityBins)

        # Calculating the "Charged" <pT> values
        for i in range(10):
            sum = dNdy[i, 0] + dNdy[i, 1] + dNdy[i, 2]
            pion_weight = dNdy[i, 0] * pT[i, 0]
            kaon_weight = dNdy[i, 1] * pT[i, 1]
            proton_weight = dNdy[i, 2] * pT[i, 2]
            pT[i, 3] = (pion_weight + kaon_weight + proton_weight) / sum

        # Calculating the Error
        # Sectioning the data into 10 equal sized groups. Calculating the pT for each group, and taking the std of mean
        SampleSize = int((len(Event['initial_entropy']) - (len(Event['initial_entropy']) % 10)) / 10)
        # Particle      Sample #    10 values

        SampledValues = np.ones([4, 10, 10])
        for i in range(10):
            # Sampling the <pT>
            pTSamplePion = Event['mean_pT']['pion'][SampleSize * i: SampleSize * (i + 1)]
            pTSampleKaon = Event['mean_pT']['kaon'][SampleSize * i: SampleSize * (i + 1)]
            pTSampleProton = Event['mean_pT']['proton'][SampleSize * i: SampleSize * (i + 1)]

            # Sampling the multiplicity
            dNdySamplePion = Event['dN_dy']['pion'][SampleSize * i: SampleSize * (i + 1)]
            dNdySampleKaon = Event['dN_dy']['kaon'][SampleSize * i: SampleSize * (i + 1)]
            dNdySampleProton = Event['dN_dy']['proton'][SampleSize * i: SampleSize * (i + 1)]

            # Sampling the Entropy
            EntropySample = Event['initial_entropy'][SampleSize * i: SampleSize * (i + 1)]

            # Binning the sampled <pT>
            BinnedpTSamplePion, bin_edges, bin_number = stats.binned_statistic(EntropySample, pTSamplePion, bins=CentralityBins)
            BinnedpTSampleKaon, bin_edges, bin_number = stats.binned_statistic(EntropySample, pTSampleKaon, bins=CentralityBins)
            BinnedpTSampleProton, bin_edges, bin_number = stats.binned_statistic(EntropySample, pTSampleProton, bins=CentralityBins)

            # Binning the sampled dNdy
            BinneddNdySamplePion, bin_edges, bin_number = stats.binned_statistic(EntropySample, dNdySamplePion, bins=CentralityBins)
            BinneddNdySampleKaon, bin_edges, bin_number = stats.binned_statistic(EntropySample, dNdySampleKaon, bins=CentralityBins)
            BinneddNdySampleProton, bin_edges, bin_number = stats.binned_statistic(EntropySample, dNdySampleProton, bins=CentralityBins)

            # Appending the data to a nice matrix and calculating the sampled charged <pT>
            for ii in range(len(BinnedpTSamplePion)):
                sum = BinneddNdySamplePion[ii] + BinneddNdySampleKaon[ii] + BinneddNdySampleProton[ii]
                pion_weight = BinneddNdySamplePion[ii] * BinnedpTSamplePion[ii]
                kaon_weight = BinneddNdySampleKaon[ii] * BinnedpTSampleKaon[ii]
                proton_weight = BinneddNdySampleProton[ii] * BinnedpTSampleProton[ii]
                SampledValues[0, ii, i] = BinnedpTSamplePion[ii]
                SampledValues[1, ii, i] = BinnedpTSampleKaon[ii]
                SampledValues[2, ii, i] = BinnedpTSampleProton[ii]
                SampledValues[3, ii, i] = (pion_weight + kaon_weight + proton_weight) / sum

        for i in range(10):
            pT[i, 4] = np.std(SampledValues[0, i, :]) / m.sqrt(10)
            pT[i, 5] = np.std(SampledValues[1, i, :]) / m.sqrt(10)
            pT[i, 6] = np.std(SampledValues[2, i, :]) / m.sqrt(10)
            pT[i, 7] = np.std(SampledValues[3, i, :]) / m.sqrt(10)
        return pT

    def FC(self, n, CentralityBins = None, vn44 = False):
        """
        Taking the events defined in EventOutput() and calculating the flow coefficients.
        Define the value of n that you are interested in to calculate Vn{2} and Vn{4}
        Optional, externally define the centrality bins.
        """
        w.filterwarnings("ignore")
        Event = self.FullAray
        EventEntropy = Event['initial_entropy']

        if CentralityBins is None:
            CentralityBins = np.append(0, np.append(np.flip(self.Centrality()[0]), m.ceil(max(EventEntropy))))
        else:
            CentralityBins = np.append(0, np.append(CentralityBins, m.ceil(max(EventEntropy))))

        # Pulling the data from our event output
        Qn = Event['flow']['Qn'][:, n - 1]
        Q2n = Event['flow']['Qn'][:, 2 * n - 1]
        M = Event['flow']['N']

        # Isolating and removing the rows that are invalid due to a small M value
        # Creating a list of all the bad indexes
        InvalidRow = []
        for i in range(2 * n + 1):
            InvalidRow = np.append(InvalidRow, np.where(M == i))
        InvalidRow = np.sort(InvalidRow.astype(int))

        # Removing all the bad indexes from the data
        if len(InvalidRow) != 0:
            for i in range(len(InvalidRow)):
                Qn = np.delete(Qn, (InvalidRow[i] - i), 0)
                Q2n = np.delete(Q2n, (InvalidRow[i] - i), 0)
                M = np.delete(M, (InvalidRow[i] - i), 0)
                EventEntropy = np.delete(EventEntropy, (InvalidRow[i] - i), 0)

        ####################################################################################################################

        # Calculating the Cn{2} & Cn{4}
        CalculationMatrix = np.zeros([len(M), 2])
        # Cn{2} & Cn{4} See https://arxiv.org/pdf/1010.0233.pdf equation 16, and 18:
        CalculationMatrix[:, 0] = (abs(Qn) ** 2 - M) / (M * (M - 1))
        CalculationMatrix[:, 1] = ((abs(Qn) ** 4) + (abs(Q2n) ** 2) - 2 * np.real(Q2n * np.conj(Qn) * np.conj(Qn)) -
                                   2 * (2 * (M - 2) * (abs(Qn) ** 2) - (M * (M - 3)))) / (
                                              M * (M - 1) * (M - 2) * (M - 3))

        FlowCoefficients = np.ones([10, 4])
        # Vn{2}, Vn{4}, Vn{2} error, Vn{4} error

        binnedQn, bin_edges, bin_number = stats.binned_statistic(EventEntropy, CalculationMatrix[:, 0], bins=CentralityBins)
        binnedQ2n, bin_edges, bin_number = stats.binned_statistic(EventEntropy, CalculationMatrix[:, 1], bins=CentralityBins)


        if vn44:
            limited_vn44 = np.ones([10,2])
            limited_vn44[:, 0] = -(binnedQ2n - 2 * binnedQn ** 2)

        # Calculating Vn{2}
        FlowCoefficients[:, 0] = np.sqrt(binnedQn)

        # Calculating Vn{4}
        FlowCoefficients[:, 1] = np.sqrt(np.sqrt(-(binnedQ2n - 2 * binnedQn ** 2)))

        ####################################################################################################################
        """
        Calculating standard deviation of the mean for Cn{2} & Cn{4}:
            1) Break the filtered data down into 10 equal sized groups.
                Note: Because the data may not be cleanly divisible by 10, the remainder is cut off
            2) Calculate Cn{2} and Cn{4} for each subgroup
            3) Calculate stdm for Cn{2} and Cn{4}
            4) Calculate the error for the Vn{2}, Vn{4}

        Vn{2} = sqrt(Cn{2})
        ∆Vn{2} = ∆Cn{M} / (2 * Vn{M})
            ∆Cn{2}   = stdm_Qn
            Vn{2}    = flow_coefficients[:,0]

        Vn{4} = [ 2 * Cn{2} ** 2 - Cn{4} ] ** 1/4
        ∆Vn{4} = sqrt[ (Cn{2} * Vn{4} ** -3 * ∆Cn{2} ) ** 2 + ( -1/4 * Vn{4} ** -3 * ∆Cn{4} ) ** 2 ]
            Cn{2}   = binned_Q2
            Vn{4}   = flow_coefficients[:,1]
            ∆Cn{2}  = stdm_Qn

            Vn{4}   = flow_coefficients[:,1]
            ∆Cn{4}  = stdm_Q2n
            
            
        Vn{4}^4 = 2 * Cn{2} ** 2 - Cn{4}
        ∆Vn{4}^4 = sqrt[ ( 4 * Cn{2} * ∆Cn{2} )**2 + ( ∆Cn{4} ) ** 2   ]
        """
        section_size = int((len(CalculationMatrix) - (len(CalculationMatrix) % 10)) / 10)
        BinnedSectionedMatrix = np.ones((2, 10, 10))
        # N1 = Qn and Q2n    N2 = iteration      N3 = binned value
        # Matrix 1           Row                 Column
        for i in range(10):
            sectionQn = CalculationMatrix[:, 0][section_size * i: section_size * (i + 1)]
            sectionQ2n = CalculationMatrix[:, 1][section_size * i: section_size * (i + 1)]
            sectionEntropy = EventEntropy[section_size * i: section_size * (i + 1)]

            BinnedSectionedMatrix[0, i, :], bin_edges, bin_number = stats.binned_statistic(sectionEntropy, sectionQn, bins=CentralityBins)
            BinnedSectionedMatrix[1, i, :], bin_edges, bin_number = stats.binned_statistic(sectionEntropy, sectionQ2n, bins=CentralityBins)

        # Calculating the std mean for Qn & Q2n
        # This is repeated for each BINNED VALUE

        stdm_Qn = stdm_Q2n = []
        for i in range(10):
            stdm_Qn = np.append(stdm_Qn, np.std(BinnedSectionedMatrix[0, :, i]) / m.sqrt(10))
            stdm_Q2n = np.append(stdm_Q2n, np.std(BinnedSectionedMatrix[1, :, i]) / m.sqrt(10))

        # Calculating the error
        if vn44:
            limited_vn44[:, 1] = np.sqrt( (4 * FlowCoefficients[:, 0] * stdm_Qn) ** 2 +
                                          (stdm_Q2n) ** 2)
            return limited_vn44

        FlowCoefficients[:, 2] = stdm_Qn / (2 * FlowCoefficients[:, 0])
        FlowCoefficients[:, 3] = np.sqrt((binnedQn * (FlowCoefficients[:, 1] ** (-3)) * stdm_Qn) ** 2 +
                                          (0.25 * (FlowCoefficients[:, 1] ** -3) * stdm_Q2n) ** 2)

        return FlowCoefficients

    def dNdeta(self, CentralityBins = None):
        w.filterwarnings('ignore')
        Event = self.FullAray
        if CentralityBins is None:
            CentralityBins = np.flip(self.Centrality()[0])
        centrality_bin = np.append(0, np.append(CentralityBins,m.ceil(max(Event['initial_entropy']))))

        dN = np.ones([10, 2])
        dN[:, 0], BE, BN = stats.binned_statistic(Event['initial_entropy'], Event['dNch_deta'], bins=centrality_bin)

        sample_size = int((len(Event['dNch_deta']) - (len(Event['dNch_deta']) % 10)) / 10)
        samples = np.ones((10, 10))
        # Sample #    10 values

        for i in range(10):
            sample_dN = Event['dNch_deta'][sample_size * i: sample_size * (i + 1)]
            sample_entropy = Event['initial_entropy'][sample_size * i: sample_size * (i + 1)]

            samples[i, :], bin_edges, bin_number = stats.binned_statistic(sample_entropy, sample_dN, bins=centrality_bin)

        for i in range(10):
            dN[i, 1] = np.std(samples[:, i]) / m.sqrt(10)
        return dN


class TrentoOutput:
    def __init__(self, RawFiles):
        # Taking a Trento Output file, reading it specifically for it's multiplicity.
        # For other output options, see http://qcd.phy.duke.edu/trento/
        Multiplicity = []
        e2 = []
        e3 = []
        for file in RawFiles:
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    RawMultiplicity = float(np.fromstring(line, sep='\t')[3])
                    Rawe2 = float(np.fromstring(line, sep='\t')[4])
                    Rawe3 = float(np.fromstring(line, sep='\t')[5])
                    Multiplicity.append(RawMultiplicity)
                    e2.append(Rawe2)
                    e3.append(Rawe3)
        self.Multiplicity = Multiplicity
        self.e2 = np.power(e2, 2)
        self.e3 = np.power(e3, 2)

    def MultiplicityBins(self):
        MultiplicityArray = self.Multiplicity
        max_entropy = m.ceil(max(MultiplicityArray))
        centrality_matrix = np.ones([2,9])
        centrality_calculation = np.ones([3,max_entropy])

        bin_percentage = [0, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1]
        PerBinL = np.diff(bin_percentage) / 2 + np.array(bin_percentage[:(len(bin_percentage) - 1)])
        centrality_matrix[1] = [80, 70, 60, 50, 40, 30, 20, 10, 5]  # 100 * (1 - np.array(bin_percentage[1:10]))

        # Sorting the entire Multiplicity output into a histogram for each multiplicity value
        centrality_calculation[0], bins = np.histogram(MultiplicityArray, bins=max_entropy,range=(0, max_entropy), density=True)
        # Converting the histogram to a percentage of total entropy
        for i in range(max_entropy):
            centrality_calculation[1][i] = sum(centrality_calculation[0][:i + 1])
        # Sorting the entropy into sectioned bins
        centrality_calculation[2] = pd.cut(centrality_calculation[1],bins=bin_percentage, labels=PerBinL)

        DuArr = []
        for i in range(9):
            for ii in range(len(centrality_calculation[0])):
                if centrality_calculation[2][ii] == PerBinL[i]:
                    DuArr = np.append(DuArr, ii + 1)
            # recording the maximum entropy cut off region for a given bin
            centrality_matrix[0][i] = max(DuArr)
        return centrality_matrix

    def Eccentricity2(self):
        # bin the e2 based off of multiplicty
        bins = self.MultiplicityBins()[0]
        binnede2, BE, BN = stats.binned_statistic(self.Multiplicity, self.e2, bins=bins)
        SampleSize = int((len(self.Multiplicity) - (len(self.Multiplicity) % 10)) / 10)
        # Particle      Sample #    10 values

        SampledValues = np.ones([10, 10])
        for i in range(10):
            # Sampling the <pT>
            EcSample = self.e2[SampleSize * i: SampleSize * (i + 1)]
            MultSample = self.Multiplicity[SampleSize * i: SampleSize * (i + 1)]

            # Binning the sample
            BinnedpTSamplePion, bin_edges, bin_number = stats.binned_statistic(MultSample, EcSample, bins=bins)

            # Appending the data to a nice matrix and calculating the sampled charged <pT>
            for ii in range(len(BinnedpTSamplePion)):
                SampledValues[ii, i] = np.sqrt(BinnedpTSamplePion[ii])

        Eerror = []
        for i in range(10):
             Eerror.append(np.std(SampledValues[i, :]) / m.sqrt(10))

        return np.sqrt(binnede2), Eerror[:-2]

    def Eccentricity3(self):
        # bin the e3 based off of multiplicty
        bins = self.MultiplicityBins()[0]
        binnede3, BE, BN = stats.binned_statistic(self.Multiplicity, self.e3, bins=bins)
        SampleSize = int((len(self.Multiplicity) - (len(self.Multiplicity) % 10)) / 10)
        # Particle      Sample #    10 values

        SampledValues = np.ones([10, 10])
        for i in range(10):
            # Sampling the <pT>
            EcSample = self.e3[SampleSize * i: SampleSize * (i + 1)]
            MultSample = self.Multiplicity[SampleSize * i: SampleSize * (i + 1)]

            # Binning the sample
            BinnedpTSamplePion, bin_edges, bin_number = stats.binned_statistic(MultSample, EcSample, bins=bins)

            # Appending the data to a nice matrix and calculating the sampled charged <pT>
            for ii in range(len(BinnedpTSamplePion)):
                SampledValues[ii, i] = np.sqrt(BinnedpTSamplePion[ii])

        Eerror = []
        for i in range(10):
             Eerror.append(np.std(SampledValues[i, :]) / m.sqrt(10))

        return np.sqrt(binnede3), Eerror[:-2]


class HydroOutput:
    def __init__(self, RawFile, DestinyFile):
        self.file = RawFile
        self.Destiny = DestinyFile

    def AnalyzeEvolution(self):
        RawFile = self.file
        DestinyFile = self.Destiny
        dir_check(DestinyFile)
        """
        This creates 6 files to a defined file destination:
        The X_ and Y_ grid used during the evolution in /X_.npy and /Y_.npy respectively
            The X_ and Y_ grids are the same for each slice so only one slice is recorded
            
        The time steps of the evolution in /Time.npy
        
        The temperature values in /Temperature.npy
            Records the individual temperatures for each point on the grid, for each time slice
            temperature[time slice][temperature at point on grid]
            
        The energy density values in /EnergyDensity.npy
            Records the individual energy densities for each point on the grid, for each time slice
            EnergyDensity[time slice][energy density at a point on the grid]
            
        The Knudsen and Reynolds values for each point on the grid, for each time slice, for both lower and upper π in /KnudsenReynolds.npy
            KnudReyn[time slice][data type][value at a point on the grid]
            data types are Knudsen lowercase pi, Knudsen uppercase pi, Reynolds lowercase pi, reynolds Uppercase pi.
        """
        # saving the necessary data as defined
        totalEnergyDensity = []
        totalKnudsenReynolds = []
        totalTemperature = []
        totalTime = []

        slice_array = np.empty([0, 9], dtype=float)
        #   Time    X   Y   ED  knL knU reL reU temp
        #   0       1   2   3   4   5   6   7   8

        with open(RawFile) as f:
            lines = f.readlines()
            it_num = 0
            finaltime = float(np.fromstring(lines[-1], sep='\t')[0])
            print('Total evolution time: {:.2f} fm/c'.format(finaltime))
            for line in lines:
                # Extracting the data from the output file, line by line
                line_time = float(np.fromstring(line, sep='\t')[0])
                line_x = float(np.fromstring(line, sep='\t')[1])
                line_y = float(np.fromstring(line, sep='\t')[2])
                line_temp = float(np.fromstring(line, sep='\t')[3])
                line_ED = float(np.fromstring(line, sep='\t')[4])
                line_knL = float(np.fromstring(line, sep='\t')[-4])
                line_knU = float(np.fromstring(line, sep='\t')[-3])
                line_reL = float(np.fromstring(line, sep='\t')[-2])
                line_reU = float(np.fromstring(line, sep='\t')[-1])

                # Adding each line to a single matrix
                slice_array = np.vstack((slice_array,[line_time, line_x, line_y, line_ED, line_knL, line_knU, line_reL, line_reU, line_temp]))

                # If the time value changes, the reading pauses and saves the slice of data
                if len(np.unique(slice_array[:, 0])) == 2:
                    slice_time = slice_array[:, 0][1]
                    print('Currently on slice #{}: {:.2f} fm/c'.format(it_num, slice_time))

                    # appending the data for the Knudsen and Reynold values
                    KR_slice = []
                    KnLhold = slice_array[:, 4][0:-1]
                    KnUhold = slice_array[:, 5][0:-1]
                    ReLhold = slice_array[:, 6][0:-1]
                    ReUhold = slice_array[:, 7][0:-1]
                    if it_num == 0:  # The first time slice is always doubled
                        maxKRlen = int(len(KnLhold) / 2)
                        KnLhold = KnLhold[0:maxKRlen]
                        KnUhold = KnUhold[0:maxKRlen]
                        ReLhold = ReLhold[0:maxKRlen]
                        ReUhold = ReUhold[0:maxKRlen]
                    KR_slice.append(KnLhold.tolist())
                    KR_slice.append(KnUhold.tolist())
                    KR_slice.append(ReLhold.tolist())
                    KR_slice.append(ReUhold.tolist())
                    totalKnudsenReynolds.append(KR_slice)

                    # appending the temperature data
                    temp_slice = slice_array[:, 8][:-1]
                    if it_num == 0:  # The first time slice is always doubled
                        maxTemplen = int(len(temp_slice) / 2)
                        temp_slice = temp_slice[0:maxTemplen]
                    totalTemperature.append(temp_slice)

                    # appending the energy density data
                    EDslice = slice_array[:, 3][0:-1]  # /maxED
                    if it_num == 0:  # The first time slice is always doubled
                        maxEDlen = int(len(EDslice) / 2)
                        EDslice = EDslice[0:maxEDlen]
                    totalEnergyDensity.append(EDslice)

                    if it_num == 1:
                        np.save('{}'.format(DestinyFile) + '/X_.npy', slice_array[:, 1][0:-1])
                        np.save('{}'.format(DestinyFile) + '/Y_.npy', slice_array[:, 2][0:-1])

                    totalTime.append(slice_time)

                    slice_array = slice_array[-1, :]
                    it_num += 1

        np.save('{}'.format(DestinyFile) + '/EnergyDensity.npy', totalEnergyDensity)
        np.save('{}'.format(DestinyFile) + '/KnudsenReynolds.npy', totalKnudsenReynolds)
        np.save('{}'.format(DestinyFile) + '/Temperature.npy', totalTemperature)
        np.save('{}'.format(DestinyFile) + '/Time.npy', totalTime)
        return


if __name__ == '__main__':
    this_is_a_place_holder = True
    trentoPath = ['/path/to/trento']
    print(TrentoOutput(trentoPath).MultiplicityBins())
    print(TrentoOutput(trentoPath).MultiplicityBins())
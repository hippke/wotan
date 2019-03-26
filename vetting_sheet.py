import numpy
import time as ttime
import os
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import sigma_clip
from astropy.stats import LombScargle
from transitleastsquares import transitleastsquares, cleaned_array, transit_mask, \
    catalog_info, fold, resample
from transitleastsquares.helpers import running_mean_equal_length



def vetting_figure(TIC_ID, planet_number, results, t, y, y_filt, trend, rawtime, rawflux):

    fig = plt.figure(figsize=(10, 14))
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID=TIC_ID)

    # Raw flux
    axes_1a = plt.subplot(G[0:1, :2])
    plt.plot(rawtime, rawflux/numpy.mean(rawflux), "k", linewidth=0.5)
    plt.plot(rawtime, trend/numpy.mean(y), color='red', linewidth=0.5)
    plt.xlim(min(t), max(t))
    plt.ylim(numpy.percentile(rawflux, 1), numpy.percentile(rawflux, 99))
    plt.xticks(())
    plt.ylabel(r'Raw Flux')

    ppm_flux = -(1-y_filt)*10**6

    # Detrended flux
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0.4)
    axes_1b = plt.subplot(G[1:2, :2])
    plt.plot(t, ppm_flux, "k", linewidth=0.5)

    top_ppm = -(1-max(y_filt))*10**6
    bottom_ppm = -(1-min(y_filt))*10**6
    y_range = abs(bottom_ppm) + abs(top_ppm)
    x_range = max(t) - min(t)

    offset_y = bottom_ppm + y_range/40
    offset_x = min(t) + x_range/80
    #print(top_ppm, bottom_ppm, y_range, offset_y)

    std = numpy.std(ppm_flux)
    text = r'$\sigma=$' + format(std, '.0f') + r'$\,$ppm'
    #print(bottom_ppm, top_ppm, offset, offset)
    plt.text(offset_x, offset_y, text, color='red')


    plt.ylabel(r'Flux (ppm)')
    plt.xlabel('Time (BKJD, days)')
    plt.xlim(min(t), max(t))

    # Phase fold
    axes_2 = plt.subplot(G[2, 0:2])
    plt.plot((results.model_folded_phase-0.5)*results.period, -(1-results.model_folded_model)*10**6, color='red', zorder=99)
    plt.scatter((results.folded_phase-0.5)*results.period, -(1-results.folded_y)*10**6, color='black', s=2, alpha=0.5, zorder=2)
    plt.xlim(-2*results.duration, 2*results.duration)
    plt.xlabel('Time from mid-transit (days)')
    plt.ylabel('Flux')
    plt.ylim((numpy.percentile(-(1-y_filt)*10**6, 0.1), numpy.percentile(-(1-y_filt)*10**6, 99.5)))
    plt.text(-1.95*results.duration, numpy.percentile(-(1-y_filt)*10**6, 0.1), 'primary')
    plt.plot(
        (results.folded_phase-0.5)*results.period,
        -(1-running_mean_equal_length(results.folded_y, 10))*10**6,
        color='blue', linestyle='dashed', linewidth=1, zorder=3)

    # Phase fold to secondary eclipse
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0.3)
    axes_3 = plt.subplot(G[2, 2])
    plt.yticks(())
    phases = fold(time=t, period=results.period, T0=results.T0)
    sort_index = numpy.argsort(phases, kind="mergesort")
    phases = phases[sort_index]
    flux = y_filt[sort_index]
    plt.scatter((phases-0.5) * results.period, flux, color='black', s=2, alpha=0.5, zorder=2)
    plt.plot((-0.5, 0.5), (1, 1), color='red')
    plt.xlim(-2*results.duration, 2*results.duration)
    plt.ylim(numpy.percentile(y_filt, 0.1), numpy.percentile(y_filt, 99.5))
    plt.plot(
        (phases-0.5) * results.period,
        running_mean_equal_length(flux, 10),
        color='blue', linestyle='dashed', linewidth=1, zorder=3)
    # Calculate secondary eclipse depth
    intransit_secondary = numpy.where(numpy.logical_and(
        (phases-0.5)*results.period > (-0.5*results.duration),
        (phases-0.5)*results.period < (0.5*results.duration)))
    mean = -(1 - numpy.mean(flux[intransit_secondary]))*10**6
    stabw = -(numpy.std(flux[intransit_secondary]) / numpy.sqrt(len(flux[intransit_secondary]))) * 10**6
    significance_eb = mean / stabw

    #print(mean, stabw, significance)
    plt.scatter((phases[intransit_secondary]-0.5)* results.period, flux[intransit_secondary], color='orange', s=20, alpha=0.5, zorder=0)
    if numpy.isnan(mean):
        mean = 0
    if numpy.isnan(stabw):
        stabw = 0
    if numpy.isnan(significance_eb):
        significance_eb = 99
    text = r'secondary ' + str(int(mean)) + r'$\pm$' + str(int(stabw)) + ' ppm (' + format(significance_eb, '.1f') + r'$\,\sigma$)'
    plt.text(-1.95*results.duration, numpy.percentile(y_filt, 0.1), text)
    print('secondary vespa', (mean-stabw)*10**-6)
    # Full phase
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=1)
    axes_5 = plt.subplot(G[3, :])
    plt.plot(results.model_folded_phase, -(1-results.model_folded_model)*10**6, color='red')
    plt.scatter(results.folded_phase, -(1-results.folded_y)*10**6, color='black', s=2, alpha=0.5, zorder=2)
    plt.xlim(0, 1)
    plt.ylim(numpy.percentile(ppm_flux, 0.1), numpy.percentile(ppm_flux, 99.9))
    plt.xlabel('Phase')
    plt.ylabel('Flux')

    # Check if phase gaps
    phase_diffs = abs(results.folded_phase - numpy.roll(results.folded_phase, -1))
    phase_diffs = phase_diffs[:-1]
    largest_phase_peak = max(phase_diffs) / numpy.median(phase_diffs)

    # All transits in time series
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    axes_6b = plt.subplot(G[4, :])
    y_filt = -(1-y_filt)*10**6
    in_transit = transit_mask(t, results.period, results.duration, results.T0)

    t_diff1 = abs(t - numpy.roll(t, -1))
    t_diff2 = abs(t - numpy.roll(t, +1))
    if max(t_diff1[in_transit]) > 0.1 or max(t_diff2[in_transit]) > 0.1:  # in days
        transit_near_gaps = True
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: Transits near gaps')
    else:
        transit_near_gaps = False

    transit_touches_edge = False
    if max(t) in t[in_transit]:
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: Last transit touches end')
        transit_touches_edge = True
    if max(t) in t[in_transit]:
        plt.text(min(t), numpy.percentile(ppm_flux, 0.1), 'Warning: First transit touches start')
        transit_touches_edge = True
    plt.scatter(t[in_transit], y_filt[in_transit], color='red', s=2, zorder=0)
    plt.scatter(t[~in_transit], y_filt[~in_transit], color='black', alpha=0.5, s=2, zorder=0)
    plt.plot(results.model_lightcurve_time, -(1-results.model_lightcurve_model)*10**6, alpha=0.5, color='red', zorder=1)
    plt.xlim(min(t), max(t))
    plt.ylim(numpy.percentile(ppm_flux, 0.1), numpy.percentile(ppm_flux, 99.9))
    plt.xlabel('Time (BKJD, days)')
    plt.ylabel('Flux')
    #plt.xticks(())

    # Transit depths error bars
    avg = -(1-results.depth_mean[0])*10**6
    if numpy.isnan(results.transit_depths_uncertainties).any():
        step = 0
    else:
        step = max(results.transit_depths_uncertainties)*10**6
    down = avg - step
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=1)
    axes_6 = plt.subplot(G[5, :])
    plt.errorbar(
        results.transit_times,
        -(1-results.transit_depths)*10**6,
        yerr=step,
        fmt='o',
        color='red')
    plt.plot((min(t), max(t)), (0, 0), color='black')
    plt.plot((min(t), max(t)), (avg, avg), color='black', linestyle='dashed')
    plt.xlim(min(t), max(t))
    for transit in range(len(results.transit_times)):
        plt.text(
            results.transit_times[transit],
            down,
            str(int(results.per_transit_count[transit])),
            horizontalalignment='center')
        #print(str(int(results.per_transit_count[transit])))
    plt.xlabel('Time (BKJD, days)')
    plt.ylabel('Flux')

    # Test statistic
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    #G.update(hspace=3)
    axes_7 = plt.subplot(G[6, :])
    axes_7.axvline(results.period, alpha=0.4, lw=3)
    for n in range(2, 5):
        axes_7.axvline(n*results.period, alpha=0.4, lw=1, linestyle="dashed")
        axes_7.axvline(results.period / n, alpha=0.4, lw=1, linestyle="dashed")
    plt.ylabel(r'TLS SDE')
    plt.xticks(())
    plt.plot(results.periods, results.power, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.text(results.period + 1, results.SDE * 0.9, 'SDE=' + format(results.SDE, '.1f'))

    # LS periodogram raw
    #freqs, lspower = LombScargle(rawtime, rawflux).autopower()
    freqs = numpy.geomspace(1/min(results.periods), 1/max(results.periods), 10000)
    lspower = LombScargle(rawtime, rawflux).power(freqs)
    freqs = 1/freqs
    #for i in range(len(freqs)):
    #    print(freqs[i], lspower[i])
    #print('max(results.periods)', max(results.periods))
    #idx_end = numpy.argmax(freqs<max(results.periods))
    #print(lspower)
    #print('idx_end', idx_end)
    #print('idx_end2', numpy.argmax(freqs<5))
    #freqs = freqs[:idx_end]
    #lspower = lspower[:idx_end]
    #print(lspower)
    #continuum = numpy.std(lspower)
    #lspower = numpy.divide(lspower, continuum)  # Normalize to S/N
    #print(lspower)

    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    #G.update(hspace=0)
    axes_8 = plt.subplot(G[7, :])
    axes_8.axvline(results.period, alpha=0.4, lw=15)
    #axes_8.set_yscale("log")

    plt.ylabel(r'LS (raw)')
    plt.plot(freqs, lspower, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.ylim(0, max(lspower)*1.1)
    plt.xlabel('Period (days)')
    #idx_start = numpy.argmax(freqs>results.period-2)
    #idx_end = numpy.argmax(freqs>results.period+2)
    """
    lspower_at_period = max(lspower[idx_start:idx_end])
    plt.text(results.period+1, max(lspower)*0.9, 'LS power=' + format(lspower_at_period, '.1f'))
    if lspower_at_period > results.SDE:
        text = 'sinusoidal model preferred'
    else:
        text = 'transit model preferred'
    plt.text(results.period+1, max(lspower)*0.8, text)
    """

    # Calculate Lomb-Scargle periodogram detrended
    freqs = numpy.geomspace(1/min(results.periods), 1/max(results.periods), 10000)
    lspower = LombScargle(t, y_filt).power(freqs)
    freqs = 1/freqs

    #freqs, lspower = LombScargle(t, y_filt).autopower()
    #idx_end = numpy.argmax(freqs>max(results.periods))
    #freqs = freqs[:idx_end]
    #lspower = lspower[:idx_end]
    continuum = numpy.std(lspower)
    lspower = numpy.divide(lspower, continuum)  # Normalize to S/N

    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0)
    #G.update(hspace=3)
    axes_9 = plt.subplot(G[8, :])
    peak_index = numpy.argmax(lspower)
    #print(peak_index, freqs[peak_index])
    axes_9.axvline(freqs[peak_index], alpha=0.4, lw=10)
    plt.ylabel(r'LS (detrended)')
    plt.plot(freqs, lspower, color='black', lw=0.5)
    plt.xlim(0, max(results.periods))
    plt.ylim(0, max(lspower)*1.1)
    plt.xlabel('Period (days)')

    #plt.text(freqs[peak_index]+1, max(lspower)*0.9, 'LS power=' + format(max(lspower), '.1f'))
    #if max(lspower) > results.SDE:
    #    text = 'sinusoidal model preferred'
    #    model_preferred = 'sine'
    #else:
    #    text = 'transit model preferred'
    #    model_preferred = 'transit'
    #plt.text(results.period+1, max(lspower)*0.8, text)


    # Text
    G = gridspec.GridSpec(9, 3)
    G.update(hspace=0, wspace=0.1)
    axes_8 = plt.subplot(G[0:2, 2])
    plt.xticks(())
    plt.yticks(())
    plt.text(0, 0, 'TIC ' + str(TIC_ID) + '.0' + str(planet_number), fontweight='bold')


    from astroquery.mast import Catalogs
    tess_mag = Catalogs.query_criteria(catalog="Tic", ID=TIC_ID).as_array()[0][60]

    #result = Vizier(columns=["Kpmag", "Dist"]).query_constraints(ID=TIC_ID, catalog="IV/34/epic")[0].as_array()
    #Kpmag = result[0][0]
    #Dist = result[0][1]
    Dist = 0
    plt.text(0, -0.15, 'TICmag=' + format(tess_mag, '.1f') + ', d=' + format(Dist, '.0f') + ' pc')


    plt.text(0, -0.30, 'SDE=' + format(results.SDE, '.1f') + ', SNR=' + format(results.snr, '.1f'))
    plt.text(0, -0.45, 'P=' + format(results.period, '.5f') + ' +-' + format(results.period_uncertainty, '.5f') + r'$\,$d')
    plt.text(0, -0.60, r'$T_{dur}=$' + format(results.duration, '.5f') + r'$\,$d')
    plt.text(0, -0.75, r'$R_*=$' + format(radius, '.2f') + ' (+' + format(radius_max, '.2f') + ', -' + format(radius_min, '.2f') + ') $\,R_{\odot}$')
    plt.text(0, -0.90, r'$M_*=$' + format(mass, '.2f') + ' (+' + format(mass_max, '.2f') + ', -' + format(mass_min, '.2f') + ') $\,M_{\odot}$')
    plt.text(0, -1.05, r'$R_P/R_*=$' + format(results.rp_rs, '.3f'))

    print('rp_rs', results.rp_rs)

    rp = results.rp_rs * radius  # in solar radii
    sun = 695700
    earth = 6371
    jupi = 69911
    rp_earth = (sun / earth) * rp
    rp_jupi = (sun / jupi) * rp
    plt.text(0, -1.20, r'$R_P=$' + format(rp_earth, '.2f') + '$\,R_{\oplus}=$'+ format(rp_jupi, '.2f') + '$\,R_{Jup}$')
    plt.text(
        0, -1.35,
        r'$\delta=$' + \
        format((1-results.depth_mean[0])*10**6, '.0f') + ' ppm (' + \
        format((1-results.depth_mean_odd[0])*10**6, '.0f') + ', ' + \
        format((1-results.depth_mean_even[0])*10**6, '.0f') + ')')
    plt.text(0, -1.50, r'odd/even mismatch: ' + format(results.odd_even_mismatch, '.2f') + '$\,\sigma$')
    plt.text(0, -1.65, str(results.distinct_transit_count) + '/' + str(results.transit_count) + ' transits with data')
    plt.xlim(-0.1, 2)
    plt.ylim(-2, 0.1)
    axes_8.axis('off')
    plt.subplots_adjust(hspace=0.4)


    # Additional vetting criteria
    valid = True

    #if largest_phase_peak > 10:
    #    valid = False
    #    print('Vetting fail! largest_phase_peak > 10 median cadences: ', largest_phase_peak)

    if abs(significance_eb) > 3:
        valid = False
        print('Vetting fail! significance_eb > 3')

    #if numpy.isnan(results.odd_even_mismatch):
    #    valid = False
    #    print('Vetting fail! numpy.isnan(results.odd_even_mismatch)')

    if abs(results.odd_even_mismatch) > 3:
        valid = False
        print('Vetting fail! odd_even_mismatch larger 3 sigma')

    if transit_near_gaps and results.distinct_transit_count < 5:
        valid = False
        print('Vetting fail! Transit near gaps and distinct_transit_count < 5')

    #if model_preferred != 'transit':
    #    valid = False
    #    print('Sinusoidal model was preferred over transit model')
    #if max(lspower) > results.SDE:
    #    valid = False
    #    print('Vetting fail! Sinusoidal model preferred')


    #if transit_touches_edge:
    #    valid = False
    #    print('Vetting fail! First or last transit touches edge of time series')

    valid = True  # for now to check them all
    if valid:
        figure_out_path = str(TIC_ID) + '_0' + str(planet_number-1)
        plt.savefig(figure_out_path + '.pdf', bbox_inches='tight')
        #plt.savefig(figure_out_path + '.pdf', bbox_inches='tight')
        print('Figure made:', figure_out_path)
        """
        print('Creating full CSV result file')
        numpy.savetxt(
            fname=str(EPIC_id) + '_statistics.csv',
            X=numpy.array([EPIC_id, results.period, results.T0]),
            delimiter=',',
            newline=" ",
            fmt='%.5f',
            )
        """
    else:
        print('Vetting criteria failed! No figure made.')
    #plt.close()
    print('T0', results.T0)

    return valid


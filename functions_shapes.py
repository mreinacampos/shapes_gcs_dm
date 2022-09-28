""" 

Functions required to identify isodensity contours in images
Author: Marta Reina-Campos

"""

import os, sys, numpy, scipy
sys.path.append("../")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


def create_gcs_image(x, y, img_bins, weights):
    """ Use numpy.histogram2d to produce a GC density map - units: [weight units] kpc^-2 """

    # calculate the width of a pixel - units: kpc
    width_per_pixel = numpy.diff(img_bins)[0] 
    # calculate their area
    area = numpy.power(width_per_pixel, 2)
    # create the 2d histogram
    img, xedges, yedges = numpy.histogram2d(x, y, bins = img_bins,
                                            weights = weights)
    # divide the values by the area of a pixel -> kpc^-2
    img /= area
    img = img.transpose()
    # return the image
    return img


def determine_contours_long_segment(cs, do_component = "None"):
    """ Given a set of contours at different levels, determine & keep the longest at each level """
    dict_cs = {}
    # already in the right order
    if "star" in do_component:
        cs_levels = range(len(cs.levels))
    else: # require inversion
        cs_levels = range(len(cs.levels))[::-1]
        
    # for each level of the contours
    for ind in cs_levels:
        # determine the number of segments found at this level
        num_elements = len(cs.allsegs[ind])
        idx_longseg = -1
        num_longseg = 0
        # for each segment
        for i in range(num_elements):
            
            # is it the longest one?
            if len(cs.allsegs[ind][i]) > num_longseg:
                idx_longseg = i
                num_longseg = len(cs.allsegs[ind][i])

            # once we have the longest one, save it!
            if i == num_elements - 1:
                # save the coordinates of the longest element in each level 
                if "star" in do_component:
                    ind_cs = cs_levels[-1] - ind
                else:
                    ind_cs = ind
                dict_cs[ind_cs] = numpy.asarray([cs.allsegs[ind][idx_longseg][:,0],
                                              cs.allsegs[ind][idx_longseg][:,1]]).T
                
    return dict_cs



def identify_isodensity_contours(img, rgclim, rad_rrgclim_contours, do_component = "star", 
                                 do_img = True, return_contours = True, return_rad_profile = True,
                                 annotation = "", postfix = "", cut_surfbright = 28):
    """
    Function that identifies isodensity contours in image 
    Input:
        * img: image in which to identify the isodensity contours
        * rgclim: outer edge of the image (width = 2*rgclim)
        * rad_rrgclim_contours: radius at which to draw the contours [in units of rgclim]
        * do_component: ["star" or "gcs"] each component predefines some variables
        * saveimgs [boolean]: whether the image should be saved 
        * return_contours [boolean]: whether to return the longest contour per level
        * return_rad_profile [boolean]: whether to return the radial profile
        * annotation [str]: string to be annotated in the right-most panel
        * postfix [str]: string to be used as postfix of the image
    """
    
    # predefine variables based on the component (e.g. colorbar limits, colourmaps, ...)
    if "star" in do_component:
        cmin = 16; cmax = cut_surfbright
        cbar_title = r"$\mu_{\rm r-band}$ [$ \rm mag/arcsec^{2}$]"
        img_cmap = plt.cm.Greys_r
        fname = "plot_contours_star_surfbright{:s}"
        norm = None
    elif "gcs" in do_component:
        cmin = 1e-4; cmax = 5e0 
        img += cmin*0.0001
        cbar_title = r"GC number density [$ \rm kpc^{-2}$]"
        img_cmap = plt.cm.Oranges
        fname = "plot_contours_gcs_gmagcut_number{:s}"
        norm = LogNorm()
    else:
        print("The wrong component was specified")

    if do_img:
        # create the figure environment
        fig, ax = plt.subplots(1,3, figsize= (24,6.))
        ax = numpy.atleast_1d(ax)
        ax = ax.ravel()
    else:
        fig, ax = plt.subplots(1,1)
        ax = numpy.atleast_1d(ax)

    # number of pixels in the image
    num_pixels = img.shape[0]
    # define the radial bins needed for the radial profile - units: kpc and relative to rgclim
    radial_bins = numpy.logspace(numpy.log10(0.003*rgclim), numpy.log10(rgclim), 32+1)
    cen_radial_bins = numpy.sqrt(radial_bins[:-1]*radial_bins[1:]) # units: kpc
    cen_rrgclim = cen_radial_bins/rgclim # units: relative to rgclim    
    
    # create a discrete colormap for the radial bins
    colors = plt.cm.YlGnBu(numpy.linspace(0,1,21))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('name', colors)
    mpl.cm.register_cmap(name='YlGnBu_new', cmap=cmap)
    cmap = plt.get_cmap('YlGnBu_new', len(cen_radial_bins))
    colours = cmap(numpy.linspace(0,len(cen_radial_bins),len(cen_radial_bins)+1)/float(len(cen_radial_bins)))
    
    # create a discrete colormap for the contours
    colors_contours = plt.cm.viridis_r(numpy.linspace(0,1,21))
    cmap_contours = mpl.colors.LinearSegmentedColormap.from_list('name', colors_contours)
    mpl.cm.register_cmap(name='contours_new', cmap=cmap_contours)
    cmap_contours = plt.get_cmap('contours_new', len(rad_rrgclim_contours))
    colours_contours = cmap_contours(numpy.linspace(0,1,len(rad_rrgclim_contours)))

    if do_img:
        ### 1st panel: show the original image
        s = ax[0].imshow(img, origin='lower', cmap=img_cmap, 
                                 norm = norm, vmin = cmin, vmax = cmax,
                                extent=[-rgclim, rgclim, -rgclim, rgclim])

    ### 2nd panel: calculate the RADIAL PROFILE
    # create a grid with radii for the masking
    x = numpy.linspace(0, num_pixels, num_pixels)
    y = numpy.linspace(0, num_pixels, num_pixels)
    width_per_pixel = 2*rgclim / (num_pixels) # kpc

    x -= num_pixels/2; y -= num_pixels/2
    X, Y = numpy.meshgrid(x, y)
    X *= width_per_pixel; Y *= width_per_pixel # kpc
    rad = numpy.sqrt(numpy.power(X, 2) + numpy.power(Y, 2)) # kpc

    # loop over the radial bins and calculate the profile
    rad_profile = numpy.zeros(shape=len(radial_bins)-1)
    rad_profile_per5th = numpy.zeros(shape=len(radial_bins)-1)
    rad_profile_per95th = numpy.zeros(shape=len(radial_bins)-1)
    for ind_rad in range(len(radial_bins)-1):
        if do_img:
            # draw the radial bins in the left-most panel
            color = cmap(ind_rad/len(cen_radial_bins))
            xx = numpy.linspace(-radial_bins[ind_rad+1], radial_bins[ind_rad+1], 51)
            yy = numpy.sqrt(numpy.power(radial_bins[ind_rad+1], 2) - numpy.power(xx, 2))
            ax[0].plot(xx, yy, ls = ":", color = color, marker = "None")
            ax[0].plot(xx, -yy, ls = ":", color = color, marker = "None")

        # select the pixels within the radial bin
        mask = (img > 0)*(rad >= radial_bins[ind_rad])*(rad < radial_bins[ind_rad+1]) 
        # calculate the median density within the bin (and the 5-95th percentiles)
        if numpy.sum(mask):
            rad_profile[ind_rad] = numpy.median(img[mask])
            rad_profile_per5th[ind_rad] = numpy.percentile(img[mask], 5)
            rad_profile_per95th[ind_rad] = numpy.percentile(img[mask], 95)
            
        if do_img:
            # plot the profile in the middle panel
            ax[1].plot(cen_rrgclim[ind_rad], rad_profile[ind_rad], marker = "o", ls = "", c = color)
        
    if do_img:
        # and the percentiles
        ax[1].fill_between(cen_rrgclim, rad_profile_per5th, rad_profile_per95th, alpha = 0.2)
    
    ### 3rd panel: identify the CONTOURS
    if do_img:
        # plot the original image 
        s = ax[2].imshow(img, origin='lower', cmap=img_cmap, 
                                 norm = norm, vmin = cmin, vmax = cmax,
                                extent=[-rgclim, rgclim, -rgclim, rgclim])
    #try:
    # need to interpolate the radial profile to get the levels
    intp_func = scipy.interpolate.interp1d(cen_radial_bins, rad_profile, kind='nearest')

    # values are already in increasing order
    if "star" in do_component:
        levels = intp_func(rad_rrgclim_contours*rgclim)
        colors = colours_contours[::-1]
    else:
        levels = intp_func(rad_rrgclim_contours*rgclim)[::-1]
        colors = colours_contours

    # do not include the pixels that have been saturated
    if "gc" in do_component:
        mask_levels = levels >= cmin
    elif "star" in do_component:
        mask_levels = levels <= cmax 
    else:
        mask_levels = levels >= 0

    # obtain the contours
    if do_img:
        axis = ax[2]
    else:
        axis = ax[0]
    img_contours = axis.contour(X, Y, img,
                                 levels = levels[mask_levels],
                                 colors = colors[mask_levels])
    #except:
    #    print("Contours didn't work!")
    #    img_contours = None
        
    if do_img:
        # adjust the figure
        fig.subplots_adjust(left = 0.1, top = 0.95, bottom = 0.1, wspace = 0.3, hspace = 0.15, right=0.85)
        # set the colorbars using the positions of the axes
        axpos_bottom = ax[-1].get_position(); axpos_top = ax[-1].get_position() 
        cbar_axis = fig.add_axes([axpos_bottom.x1+0.01, axpos_bottom.y0, 0.01, axpos_top.y1 - axpos_bottom.y0])

        cb = fig.colorbar(s, cax=cbar_axis)
        if "star" in do_component:
            cb.ax.invert_yaxis()
            cb.set_ticks(numpy.arange(cmin, cmax+1, 2))
            cb.set_ticklabels(numpy.arange(cmin, cmax+1, 2))
        cb.set_label(cbar_title)

        # format all axes
        for ind in range(len(ax)):
            ax[ind].tick_params(bottom = True, left= True,
                                right = True, top = True, axis = "both", which = "both")

            if ind % 2 == 0:
                ax[ind].set_xlabel("$x$ [kpc]")
                ax[ind].set_ylabel("$y$ [kpc]")
                ax[ind].set_xlim(-rgclim, rgclim)
                ax[ind].set_ylim(-rgclim, rgclim)
                ax[ind].set_aspect("equal")

            if ind == 1:
                ax[ind].set_xscale("log")
                ax[ind].set_xlabel(r"$R/r_{200}$")
                ax[ind].set_ylabel(cbar_title)
                ax[ind].set_xlim(radial_bins[0]/rgclim, radial_bins[-1]/rgclim)

                if "star" in do_component:
                    ax[ind].set_ylim(cmax, cmin)
                else:
                    ax[ind].set_yscale("log")
                    ax[ind].set_ylim(cmin, cmax)

                # add vertical lines with the radii at which we draw the contours
                for i in range(len(rad_rrgclim_contours)):
                    color = colours_contours[(len(rad_rrgclim_contours) - i - 1)]
                    ax[ind].axvline(rad_rrgclim_contours[i],
                                    ls = "-", lw = 3, color = color)
            if ind == 2:
                # add a circle with rgclim
                xx = numpy.linspace(-rgclim, rgclim, 201)
                yy = numpy.sqrt(rgclim*rgclim - xx*xx)
                ax[ind].plot(xx, yy, ls = ":", c = "k", lw = 0.5, marker = "None")
                ax[ind].plot(xx, -yy, ls = ":", c = "k", lw = 0.5, marker = "None")

        ax[-1].annotate(annotation, xy = (0.98, 0.05), color = "k",
                               xycoords = "axes fraction", ha = "right", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=1))
        # save the figure
        filename = os.path.join(".", fname.format(postfix)+".pdf")
        fig.savefig(filename, bbox_inches='tight')
        print("Saving figure as {:s}".format(filename))

        plt.show()
    else:
        plt.close()
    
    if img_contours:
        # only return the longest contour at each level
        img_contours_long_segment = determine_contours_long_segment(img_contours, do_component = do_component)
        return img_contours_long_segment, rad_profile
    



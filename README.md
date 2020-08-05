### MP_PHOT

##### A new bulldozer-enabled workflow for minor planet photometry.

_Eric Dose, Albuquerque, NM_

_July-September 2020_

_____

Minor planet / asteroid photometry has a number of difficulties not encountered 
in variable star photometry. Almost all of the new difficulties arise from one simple fact:

**YOUR TARGETS MOVE.** 

Image by image. Night by night. You will _never_
image a given MP in exactly the same sky position twice: not after 5 minutes, 
not in 5 lifetimes. Never happen.

    The new "bulldozer" algorithm in this package is designed to remove the star background
    from images. Once a session of images (one field of view, one night) is bulldozed,
    in most cases you will see the target MP with constant PSF (shape) moving across a
    field of view with no stars. NO stars. Even if your scope points badly, tracks badly,
    or changes focus slightly. In time, it will work even if your focal length changes
    a bit during the night (still working on that one).


This rapid movement across the sky means that for MP photometry:

* Planning your observing night:
   - you have to make sure the MP is going to be in your camera's field of view.
   And a typical MP moves about 1/3 to 1/2 of a typical field of view per night.
   So a given sky position (with guide star if you need it) is only good for 2-3 nights.
   - you really should look at a high-end planetarium program (ideally with Digital
   Sky Survey overlays, as in TheSkyX) to make sure your MP isn't going to run right
   in front of bright stars.
   - Make sure you know what hours of the night each MP is available to your scope.
   - Make sure you know how bright the MP is likely to be, at least on average. You will,
   after all, have to decide an exposure time. The Minor Planet Center's ephemeris 
   services are your best friend for such data. 
     
* Image processing:

   - once you've taken your images, you will have to know which 
   light source in your image is the minor planet. 
   Not as easy as it sounds--many of my images have 6000-8000 dots.
   Happily, over a given night, the vast majority of MPs move essentially linearly
   in RA and Declination. So my approach is to use Astrometrica software to be 
   *&$% sure I'm measuring the right dot in at least
   2 images well separated in time, and then I can interpolate the MP's position in
   the rest of the night's images. Works a treat.
   - in every image, you have to make sure there are no important stars
   lurking behind or just next to the MP to mess up the measurements of MP light flux.
   And if you're looking at low-amplitude lightcurves (say, < 0.15 magnitudes), you cannot
   afford background stars need to be 3.5-4 magnitudes fainter than your MP. For a 15th
   magnitude MP, that means no stars brighter than 18.5-19 magnitude in the MP's path.
   Good luck with that.
   - BUT! it's not like most of those background stars change all that much over a
    few hours. And galaxies and other sources probably don't change at all. And as the MP
    moves, it uncovers as many stars as it covers. AHA! 
    
So one obvious approach to removing the effects of background stars is to have some software
present an image, the observer clicks on stars he sees, and the software does its best to 
subtract those from the background as the MP moves over them, image by image. And Canopus
software can do just that.

But what about less observer-centric approaches? Say, what if Sextractor or some other 
source detection algorithm automatically discovered the stars for background subtraction.
That could work. But it will subtract only coherent sources it finds, which means that 
faint stars will be missed, and probably galaxies, nebula etc etc. Better, but not great.

BUT (and here's the **bulldozer** approach): what if you aligned the images, masked out
the MP from each, averaged these MP-masked images to give a very good background image,
and subtracted this averaged background image from each of the aligned images? In the ideal
case, you would have the MP moving across a blank background.

And it works!

Well, sort of. So far. 

##### Odd REQUIREMENTS of the BULLDOZER light-flux measurement algorithm:

- The bulldozer algorithm requires aligning the images so closely it makes your teeth hurt.
In developing this very, very tight image-image alignment, I had to invent a 
new unit of distance: the millipixel. For my very typical CCD camera, a
millipixel--1/1000 of a pixel's width, is 9 millimicrons, or about the diameter of a small
protein molecule. Happily, careful convolution with a small kernel can get me image-image
 alignment within about 10 millipixels. But so far only at image centers. If the focal
 length changes during the night--and it will--you might align the image centers perfectly
 but still suffer misalignment everywhere else in the images. 
 I think I know how to beat this; it's next on my development list.
- The algorithm requires convolving every image to have the same PSF. This was a surprise.
The reason for this very odd requirement: if the different images have different PSFs (star
shapes, from the scope wiggling in the wind, or whatever), subtraction really won't work,
at least not as well as we want. For example, if one image has stars stretched slightly 
North-South, and another image has stars stretched East-West, then even if the averaged
background comes out nice and circular (not guaranteed), then when the background is 
subtracted from the first image, there will be a bit of positive flux remaining on the 
north and south sides, and a bit of negative flux on the east and west sides. 
And as the MP moves across this freckled background, it will pick up little positive and
negative flux components, that is, noise in the measured flux and thus in the lightcurve.
Not what we want.
- The algorithm requires that the flat background (between the stars) and the source background
(caused by the stars themselves) be fitted separately to each MP-masked image. 
This was not a surprise, as photometric images often do have slight (~1-2%) background and
foreground flux variations ("cirrus effect"). But it was a disappointment, and a complex
regression approach takes care of that.

It turns out that getting uniform PSFs and correcting for cirrus effect are not especially
difficult to code (thanks to the wonderful packages astropy, ccdproc, and photutils).

It turns out that **_practically all the difficulty is in getting perfect alignment_** of the 
images so that even bright stars can be subtracted away with no residual flux.

The current test platform can reduce the effect of background stars by at least 2-2.5 
magnitudes. Good but not great.

With improved extreme image alignment approaches now in development, we should get 3.5-4 
magnitudes of background suppression, that is, down to maybe 1/4 of current. 

Unlikely to get better
than that, as some stars are variable, there will be differential extinction effect from
differential star colors, and there's no guarantee that the sky background shape is the
same from image to image, especially with Clear and similar filters, and with the drive to 
observe at low altitudes, as needed for maximum time and phase coverage per night.

Well, that's where things sit as of today (August 4 2020).

Cheers, Eric
   
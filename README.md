# RDDS
Techniques for lifetime measurements using the Recoil Distance Doppler Shift method

This repository embraces several methods for the analysis of lifetimes
using differential plunger setups. Specifically, it has been developed
for the analysis of the GANIL experiment E706, aimed to measure
lifetimes in a broad region of the nuclear chart, centered at 100Zr.

## DECAY
With the RDDS method, the fraction of decays of an excited level is
measured at several points in time, spaned over few picoseconds upon 
the beam impinging the target. This allows to infer the decay
pattern for the excited level and determine its lifetime.
For that to happen, the method relay on a degrader foil, placed at 
a precise distance from the target.  
By changing the ion velocity through the foil, the gamma-rays that are 
emmited inflight will produce two distinct peaks in the energy spectrum. 
The peak separation being due to the difference in the Doppler shift.
The intensity of the peaks provides the decay fraction at the time the 
ions reach the degrader. That means, knowing the ion velocity within the 
target to degrader gap, one can relate the gap distance, measured within
micrometer precission, into flying times on the order of picoseconds.

Experiments employing RDDS techniques together with large acceptance
mass spectrometers allow measurements in a broad selection of nuclei. 
Special attention need to be placed regarding the degrader stopping
power. It needs to allow the isotopes to leave with enough energy to be
identified upstream, but at the same time it needs to be thick enough so
the Doppler difference allows to resolve the two peak components.
Additionally, thicker degraders introduce more uncertainties in the
effective plunger distance. With short-lived excited states, the amount
of gamma-rays decaying inside the degrader will produce a distribution 
in the energy spectrum between the two peak centroids.

Large acceptance mass spectrometers however, will identify the isotopes
with a broad distribution of velocities. 
 

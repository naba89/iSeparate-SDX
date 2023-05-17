# Data Augmentations
Following is a list of all the data augmentations used. 

In model forward (on GPU) (used for both MDX and CDX tracks)
- Random Mix
- Random Gain
- Channel Swap
- Sign Flip
- Zero Random Source
- Zero one random level of DWT coefficients in input mixture

In data loader (on CPU) (used only for MDX track)
- Random Time Stretch
- Random Pitch Shift
- Random Reverberation

However, not all augmentations were used for all models. 
Please check model configs for details.
# xyz-plot-grid
stable-diffusion-webui xyz-plot-grid

Check issues. I might have found an issue, hitting save saves it twice. Check on a very low count and see if it duplicates any autosaves.

For automatic1111 stable-diffusion-webui.

Place xyz_grid.py in scripts folder along side other scripts.  
Works like x/y plot, like how you would expect, but now has a z.
Works like how you'd expect it to work, with grid legends as well.

There is a kink in it, the extra labels for the z are not aligned on the grid, and the more iterations you have in z,
each iteration makes it go off by the same distance of the extra line-space. I'll try to figure it out soon.

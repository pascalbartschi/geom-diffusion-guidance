## Reproduction Steps

1. **Generate a ligand trajectory**

Generate a ligand together with the diffusion trajectory, as exemplified in:

```bash
python generate_ligands_track_diffusion.py
```

2. **Update file paths**

Edit the paths in `diffusion-animation.pml` to point to:
- the **trace-tracked ligand trajectory**, and  
- the **corresponding protein pocket**.

3. **Run the PyMOL animation**

Start PyMOL and run the script to view the current state of the animation:

```pml
run diffusion-animation.pml
```

4. **Adjust visualization**

Adapt the camera angles and the steric clash distance object in `diffusion-animation.pml` to match your ligand–pocket system.

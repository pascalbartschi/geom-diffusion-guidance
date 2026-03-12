## clean workspace and load data ##
reinit
load "C:\Users\paesc\OneDrive\docs\master_thesis\pyMOL\diffusion-process-animation\28_ligand.sdf", ligand
load "C:\Users\paesc\OneDrive\docs\master_thesis\pyMOL\diffusion-process-animation\14gs-A-rec-20gs-cbd-lig-tt-min.pdb", pocket

## setup structures ##

bg_color white
set transparency, 0.4
as surface, pocket
show sticks, pocket
color grey70, pocket and elem C
color purple, ligand and elem C
show spheres, ligand

## visualize steric clashes to show them later ##
cmd.set("state", 82, "ligand")
distance clashes_phenylalanin, (pocket and chain A and resi 8 and name CE1), (ligand and name C and index 2842+2848+2856); \
set dash_round_ends, on, clashes_phenylalanin
set dash_color, black, clashes_phenylalanin
set dash_width, 4, clashes_phenylalanin
set dash_gap, 0.3, clashes_phenylalanin

python
from pymol import cmd
cmd.delete("clash_label")
c1 = cmd.get_atom_coords("pocket and chain A and resi 8 and name CE1")
c2 = cmd.get_atom_coords("ligand and name C and index 2842")
mid = [(a+b)/2.0 for a,b in zip(c1,c2)]
mid = [mid[0]+2.75, mid[1], mid[2]+0.25]
cmd.pseudoatom("clash_label", pos=mid)
cmd.label("clash_label", repr("Steric clash threshold:\n C <-> C => 3.4 A \n(2 x 1.7 A vdW radius)"))
cmd.set("label_color","black","clash_label")
cmd.set("label_size", 56, "clash_label")
cmd.set("label_size", 56, "clashes_phenylalanin")
python end

hide dashes
hide labels
hide nonbonded
show spheres, ligand

## initialize movie ##
python
tot_states = 82
frames_per_state = 3
tot_frames = tot_states*frames_per_state
molecule_view_frames = 390
cmd.mset(f"1 x {tot_frames+molecule_view_frames}") # use pythonic version so we can use variables

cmd.mview("store", 1, state=1, object="ligand")
cmd.mview("store", tot_frames, state=tot_states, object="ligand")
cmd.mview("store", tot_frames+molecule_view_frames, state=tot_states, object="ligand")
python end

## GENERATION PROCESS: store key object frames and mdo commands ##

# frame 1
mdo 1: set sphere_scale, 2; hide dashes; hide labels; hide nonbonded; show spheres, ligand; show surface, pocket
set_view (\
     1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
     0.000000000,    0.000000000, -907.781494141,\
    28.315002441,   -2.900634766,   36.669677734,\
   622.890563965, 1192.672241211,  -20.000000000 )
mview store, 1

# frame 4
mdo 4: set sphere_scale, 7
set_view (\
     1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
    -0.000001431,   -0.000001431, -4410.617675781,\
    40.521224976,  -74.394233704,   36.669677734,\
  3645.728027344, 5175.508300781,  -20.000000000 )
mview store, 4

# frame 29
mdo 59: set sphere_scale, 6.5
#zoom ligand, -800
#mview store, 4


# frame 59
mdo 59: set sphere_scale, 6
zoom ligand, -1000
mview store, 59

# frame 64
mdo 64: set sphere_scale, 5.25

# frame 76
mdo 76: set sphere_scale, 4.6
#zoom ligand, -1200
#mview store, 76

# frame 85
mdo 85: set sphere_scale, 4
#zoom ligand, -1200
#mview store,85

# frame 88
mdo 88: set sphere_scale, 3.5

# frame 94
mdo 94: set sphere_scale, 3
#zoom ligand, -1300
#mview store, 94

# frame 98
mdo 98: set sphere_scale, 2.5

# frame 103
mdo 103: set sphere_scale, 2
#zoom ligand, -1300
#mview store, 103

# frame 108
mdo 108: set sphere_scale, 1.75

# frame 112
mdo 112: set sphere_scale, 1.5
#zoom ligand, -1350
#mview store, 112

# frame 117
mdo 117: set sphere_scale, 1.1

# frame 121
mdo 121: set sphere_scale, 0.75
#zoom ligand, -1400
#mview store, 121

# frame 130
mdo 130: set sphere_scale, 0.6
set_view (\
     1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
    -0.000000007,    0.000000041, -184.986480713,\
    30.343317032,   10.123280525,   36.669677734,\
   161.009002686,  208.963867188,  -20.000000000 )
mview store, 130

# frame 139 - 155
mdo 139: set sphere_scale, 0.5
set_view (\
     1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
    -0.000000007,    0.000000041, -184.986480713,\
    30.343317032,   10.123280525,   36.669677734,\
   161.009002686,  208.963867188,  -20.000000000 )
mview store, 139
mdo 142: set sphere_scale, 0.45
mdo 148: set sphere_scale, 0.4
mdo 154: set sphere_scale, 0.35


# frame 158
mdo 158: set sphere_scale, 0.3
set_view (\
     1.000000000,    0.000000000,    0.000000000,\
     0.000000000,    1.000000000,    0.000000000,\
     0.000000000,    0.000000000,    1.000000000,\
     0.000000477,    0.000000305,  -89.800559998,\
    29.831222534,    8.225839615,   36.669677734,\
    35.259136200,  144.341949463,  -20.000000000 )
mview store, 158
mdo 165: set sphere_scale, 0.25


# frame 185
set_view (\
     0.343373507,   -0.097077087,    0.934168398,\
     0.109505430,    0.991998076,    0.062835589,\
    -0.932793200,    0.080720440,    0.351256341,\
     0.000000089,   -0.000000312,  -62.504756927,\
    29.727922440,    8.232030869,   35.065353394,\
  -188.152267456,  313.161590576,  -20.000000000 )
mview store, 185

mdo 195: set sphere_scale, 0.225
mdo 225: set sphere_scale, 0.2




## VIEW FINAL MOLECULE: steric clashes ##

# frame 246
set_view (\
     0.328143686,   -0.097529933,    0.939579487,\
     0.110246539,    0.991812468,    0.064448722,\
    -0.938172340,    0.082436942,    0.336209357,\
     0.000004489,   -0.000001642,  -39.717437744,\
    30.573760986,    5.021485329,   33.365543365,\
  -213.538024902,  292.972930908,  -20.000000000 )
mview store, 246

# frame 261
set_view (\
     0.328143686,   -0.097529933,    0.939579487,\
     0.110246539,    0.991812468,    0.064448722,\
    -0.938172340,    0.082436942,    0.336209357,\
     0.000004489,   -0.000001642,  -40.923950195,\
    30.573760986,    5.021485329,   33.365543365,\
  -212.331497192,  294.179443359,  -20.000000000 )
mview store, 261

# frame 291
set_view (\
    -0.600238740,    0.756720841,    0.258991510,\
     0.560379624,    0.628939092,   -0.538890719,\
    -0.570682526,   -0.178329721,   -0.801568151,\
    -0.000000376,   -0.000000119, -108.472595215,\
    29.183038712,    7.933016777,   36.669677734,\
  -175.928161621,  392.873535156,  -20.000000000 )
mview store, 291

# frame 321
set_view (\
     0.148263246,    0.985419869,   -0.083444394,\
     0.963283360,   -0.124804012,    0.237709790,\
     0.223829657,   -0.115623631,   -0.967746198,\
    -0.000000376,   -0.000000119, -108.449020386,\
    29.183038712,    7.933016777,   36.669677734,\
  -175.951736450,  392.849945068,  -20.000000000 )
mview store, 321

# frame 351
set_view (\
     0.148263246,    0.985419869,   -0.083444394,\
     0.963283360,   -0.124804012,    0.237709790,\
     0.223829657,   -0.115623631,   -0.967746198,\
    -0.000067974,   -0.000030690,  -29.174077988,\
    23.656669617,    8.461778641,   37.281665802,\
  -255.232208252,  313.569610596,  -20.000000000 )
mview store, 351
mdo 341: hide surface, pocket

# frame 381
set_view (\
     0.385389328,    0.892306387,    0.235075012,\
     0.357827038,   -0.379336327,    0.853266835,\
     0.850549757,   -0.244722873,   -0.465484589,\
    -0.000002235,   -0.000042737,  -18.879205704,\
    22.368137360,    7.945026875,   35.690967560,\
  -265.530303955,  303.271514893,  -20.000000000 )
mview store, 381

mdo 396: show dashes, clashes_phenylalanin; show labels, clashes_phenylalanin

mdo 426: cmd.show("labels" ,"clash_label")

# frame 501
set_view (\
     0.385389328,    0.892306387,    0.235075012,\
     0.357827038,   -0.379336327,    0.853266835,\
     0.850549757,   -0.244722873,   -0.465484589,\
    -0.000002235,   -0.000042737,  -18.879205704,\
    22.368137360,    7.945026875,   35.690967560,\
  -265.530303955,  303.271514893,  -20.000000000 )
mview store, 501
mdo 501: cmd.hide("labels" ,"clash_label")

## ZOOM BACK ##

# frame 531
set_view (\
    -0.839561462,   -0.121353328,    0.529534340,\
    -0.167130798,   -0.869763792,   -0.464305341,\
     0.516914725,   -0.478314847,    0.709940732,\
    -0.000002235,   -0.000042737,  -18.879205704,\
    22.368137360,    7.945026875,   35.690967560,\
  -265.530303955,  303.271514893,  -20.000000000 )
mview store, 531

# frame 561
set_view (\
    -0.734773159,   -0.244424134,   -0.632741272,\
     0.655622602,   -0.495158970,   -0.570068777,\
    -0.173970371,   -0.833711743,    0.524079561,\
    -0.000002235,   -0.000042737,  -18.879205704,\
    22.368137360,    7.945026875,   35.690967560,\
  -265.530303955,  303.271514893,  -20.000000000 )
mview store, 561
mdo 586: show surface, pocket

# frame 636
set_view (\
    -0.985546052,    0.094276413,    0.140739635,\
    -0.161343545,   -0.269250542,   -0.949459791,\
    -0.051618647,   -0.958444297,    0.280570388,\
     0.000029393,   -0.000025637,  -97.674415588,\
    22.661518097,    6.179971218,   29.567470551,\
  -186.735946655,  382.065765381,  -20.000000000 )
mview store, 636













## play to debug ##

mplay